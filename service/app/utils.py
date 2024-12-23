import os
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
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