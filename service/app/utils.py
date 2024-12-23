import os
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from dvc.exceptions import DvcException
from ml_models.base_model import BaseModel

def find_model_files(models_dir):
    """
    Поиск файлов моделей в указанной директории.
    :param models_dir: Путь к директории с моделями.
    :return: Список названий файлов моделей.
    """
    model_files = []
    for file in os.listdir(models_dir):
        if file.endswith('_model.py') and file != 'base_model.py':
            model_files.append(file)
    return model_files

def import_module_from_file(file_path):
    """
    Динамический импорт модуля из файла.
    :param file_path: Путь к файлу.
    :return: Импортированный модуль.
    """
    module_name = Path(file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_model_classes(module):
    """
    Получение классов моделей из модуля.
    :param module: Импортированный модуль.
    :return: Список классов моделей.
    """
    model_classes = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, BaseModel) and attr is not BaseModel:
            model_classes.append(attr)
    return model_classes

def check_model_status(model_id, log_path='models_log.csv'):
    """
    Проверяет, существует ли запись с заданным model_id в журнале моделей и является ли модель обученной.
    
    :param model_id: Уникальный идентификатор модели.
    :param log_path: Путь к файлу журнала моделей.
    :return: bool: True - есть модель и она обучена, False иначе. 
    """
    # Проверяем, существует ли файл журнала
    if not os.path.exists(log_path):
        return False, None
    
    # Считываем файл журнала
    log_df = pd.read_csv(log_path)
    
    # Проверяем, существует ли запись с заданным model_id
    model_records = log_df[log_df['model_id'] == model_id]
    
    if model_records.empty:
        return False, None
    
    # Проверяем, является ли модель обученной
    trained = np.any(model_records['status'].map(lambda x: 'trained' in x))
    
    return True and trained, model_records['model_type'].values[0]

def upload_dataset_to_minio_and_dvc_track(minio_client, bucket_name, object_name, file_path):
    """
    Загружает датасет в MinIO и отслеживает его с помощью DVC.
    :param minio_client: Клиент Minio.
    :param bucket_name: Имя бакета.
    :param object_name: Имя объекта в бакете.
    :param file_path: Путь к файлу для загрузки.
    """
    try:
        # Загрузка файла в MinIO
        minio_client.fput_object(bucket_name, object_name, file_path)
        logger.info(f'Dataset uploaded to Minio: {object_name} from {file_path}')
        
        # Отслеживание файла с помощью DVC
        subprocess.run(['dvc', 'add', file_path], check=True)
        subprocess.run(['dvc', 'commit', '-m', f'Add dataset {object_name}'], check=True)
        logger.info(f'Dataset {object_name} tracked by DVC')
        
        # Удаление файла из буферной папки
        os.remove(file_path)
        logger.info(f'Buffer file {file_path} deleted')
        
    except NoCredentialsError:
        logger.error('Credentials not available')
        raise HTTPException(status_code=500, detail="Credentials not available")
    except ClientError as e:
        logger.error(f'Error uploading dataset to Minio: {e}')
        raise HTTPException(status_code=500, detail=f"Error uploading dataset to Minio: {e}")
    except DvcException as e:
        logger.error(f'Error tracking dataset with DVC: {e}')
        raise HTTPException(status_code=500, detail=f"Error tracking dataset with DVC: {e}")
    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

def download_dataset_from_minio(minio_client, bucket_name, object_name, file_path):
    """
    Скачивает датасет из MinIO и удаляет файл из буферной папки после использования.
    :param minio_client: Клиент Minio.
    :param bucket_name: Имя бакета.
    :param object_name: Имя объекта в бакете.
    :param file_path: Путь к файлу для сохранения.
    """
    try:
        # Скачивание файла из MinIO
        minio_client.fget_object(bucket_name, object_name, file_path)
        logger.info(f'Dataset downloaded from Minio: {object_name} to {file_path}')
                
    except NoCredentialsError:
        logger.error('Credentials not available')
        raise HTTPException(status_code=500, detail="Credentials not available")
    except ClientError as e:
        logger.error(f'Error downloading dataset from Minio: {e}')
        raise HTTPException(status_code=500, detail=f"Error downloading dataset from Minio: {e}")
    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")