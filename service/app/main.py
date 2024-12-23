from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import joblib
import pandas as pd
from datetime import datetime
from minio import Minio
import mlflow
from mlflow.models import Model
from mlflow.sklearn import save_model, load_model
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

from ml_models.catboost_model import CatBoostModel
from ml_models.linear_model import LinearModel
from ml_models.tree_model import TreeModel
from app.utils import (find_model_files, import_module_from_file, get_model_classes, check_model_status, 
    download_dataset_from_minio, upload_dataset_to_minio_and_dvc_track)
from app.pydantic_models import PredictRequest, RetrainRequest, EvaluateRequest, TrainRequest

import logging

app = FastAPI(title="ML Model Service")

# Настройка логгирования
logging.basicConfig(filename='./ml_service_logs', filemode='a', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('./ml_service_logs')

if not os.path.exists('./upload_data'):
    os.makedirs('./upload_data')

if not os.path.exists('./download_data'):
    os.makedirs('./download_data')

# Настройка MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5432"))

# Загрузка классов доступных моделей
models_dir = './ml_models'
model_files = find_model_files(models_dir)
available_models = {}

for file in model_files:
    file_path = os.path.join(models_dir, file)
    module = import_module_from_file(file_path)
    model_classes = get_model_classes(module)
    available_models[file] = model_classes

# Настройка Minio
minio_client = Minio(
    os.getenv("MINIO_ENDPOINT", "minio:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    secure=False
)

# Создание бакета в Minio
if not minio_client.bucket_exists(os.getenv("MINIO_BUCKET_NAME", "mlopsbucket")):
    minio_client.make_bucket(os.getenv("MINIO_BUCKET_NAME", "mlopsbucket"))
logger.info(f"Bucket {os.getenv('MINIO_BUCKET_NAME', 'mlopsbucket')} created successfully.")

@app.get("/models", response_model=List[str])
def get_available_models():
    """
    Возвращает список доступных для обучения классов моделей.
    """
    available_model_names = []
    for file, classes in available_models.items():
        available_model_names.extend([cls.__name__ for cls in classes])
    
    return available_model_names

@app.get("/datasets_info")
def get_available_datasets():
    """
    Возвращает список доступных для обучения датасетов.
    """
    objects = minio_client.list_objects(os.getenv("MINIO_BUCKET_NAME", "mlopsbucket"), recursive=False)
    return list(objects)

@app.get("/status", response_model=Dict[str, str])
def get_status():
    """
    Возвращает статус сервиса.
    """    
    return {"status": "OK"}

@app.get("/models_info", response_model=List[Dict[str, str]])
def get_models_info():
    """
    Возвращает информацию о моделях.
    """
    return pd.read_csv('./models_log.csv').groupby('model_id').tail(1).to_dict(orient='records')

@app.post("/upload_dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Загружает датасет в minio и отслеживает его с помощью DVC.
    """
    # Сохранение файла локально
    file_path = os.path.join('./upload_data', file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    upload_dataset_to_minio_and_dvc_track(minio_client, os.getenv("MINIO_BUCKET_NAME", "mlopsbucket"), file.filename, file_path)

    return {"filename": file.filename, "message": "Dataset uploaded and tracked successfully"}

@app.get("/download_dataset/{filename}")
async def download_dataset(filename: str):
    """
    Выгружеает датасет из minio
    """
    # Путь к файлу локально
    file_path = os.path.join('./download_data', filename)

    # Скачивание файла из Minio
    download_dataset_from_minio(minio_client, os.getenv("MINIO_BUCKET_NAME", "mlopsbucket"), filename, file_path)
    logger.info(f'Dataset downloaded from Minio: {filename} to {file_path}')

    # Читаем файл в байтах
    with open(file_path, "rb") as file:
        file_content = file.read()

    # Удаление файла из буферной папки после использования
    os.remove(file_path)
    logger.info(f'Buffer file {file_path} deleted')

    return Response(content=file_content, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})
    
@app.post("/train", response_model=Dict[str, Any])
def train_model(request: TrainRequest):
    """
    Обучает ML-модель с возможностью настройки гиперпараметров.
    """
    for file, classes in available_models.items():
        for cls in classes:
            if cls.__name__ == request.model_type:
                model = cls(model_id=None, 
                    model_description=request.model_description, 
                    model_params=request.model_params,
                    task_type=request.task_type,
                    mlflow_experiment_name = request.mlflow_experiment_name)

                # Получаем данные для обучения: 
                # Если пришел путь до файла, то скачаем из minio, если пришли данные в запросе, то сохраним датасет в minio

                if request.dataset_path is not None:
                    file_path = os.path.join('./download_data', request.dataset_path)
                    download_dataset_from_minio(minio_client, os.getenv("MINIO_BUCKET_NAME", "mlopsbucket"), filename, file_path)

                    df = pd.read_csv(file_path)
                    X = df.drop(columns='target')
                    y = df['target']

                    # Удаление файла из буферной папки после использования
                    os.remove(file_path)
                    logger.info(f'Buffer file {file_path} deleted')
                else:
                    feature_names = request.feature_names if len(request.feature_names) else None
                    X = pd.DataFrame(data=request.features, columns=feature_names)
                    y = request.target
                    df = X.copy()
                    df['target'] = y

                    filename = model.model_id + '.csv'
                    file_path = os.path.join('./upload_data', filename)
                    df.to_csv(file_path, index=False)
                    upload_dataset_to_minio_and_dvc_track(minio_client, os.getenv("MINIO_BUCKET_NAME", "mlopsbucket"), filename, file_path)

                model.train(X, y, 
                    cv=request.cv,
                    optimize_hyperparameters_flag=request.optimize_hyperparameters_flag, 
                    optimize_hyperparameters_params=request.optimize_hyperparameters_params)

                return {"model_id": model.model_id, "status": "trained"}

    raise HTTPException(status_code=404, detail=f"Model {request.model_type} not found")

@app.post("/predict")
def predict(request: PredictRequest):
    """
    Возвращает предсказание конкретной модели.
    """
    model_existing_flag, model_type = check_model_status(request.model_id)
    if not model_existing_flag:
        raise HTTPException(status_code=404, detail=f"Model with id {model_id} not found")

    for file, classes in available_models.items():
        for cls in classes:
            if cls.__name__ == model_type:
                model = cls(model_id=request.model_id)

    # Получаем данные для предикта: 
    # Если пришел путь до файла, то скачаем из minio

    if request.dataset_path is not None:
        file_path = os.path.join('./download_data', request.dataset_path)
        download_dataset_from_minio(minio_client, MINIO_BUCKET_NAME, request.dataset_path , file_path)

        df = pd.read_csv(local_dataset_path)
        if 'target' in df.columns:
            X = df.drop(columns='target')

        # Удаление файла из буферной папки после использования
        os.remove(file_path)
        logger.info(f'Buffer file {file_path} deleted')
    else:
        feature_names = request.feature_names if len(request.feature_names) else None
        X = pd.DataFrame(data=request.features, columns=feature_names)

    if request.prediction_type == 'predict_proba':
        result = model.predict_proba(X).tolist()
    else:
        result = model.predict(X).reshape(-1).tolist()
    
    return result

@app.post("/retrain", response_model=Dict[str, Any])
def retrain_model(request: RetrainRequest):
    """
    Обучает заново уже обученную модель.
    """
    model_existing_flag, model_type = check_model_status(request.model_id)
    if not model_existing_flag:
        raise HTTPException(status_code=404, detail=f"Model with id {model_id} not found")

    for file, classes in available_models.items():
        for cls in classes:
            if cls.__name__ == model_type:
                model = cls(model_id=request.model_id, 
                    model_description=request.model_description, 
                    model_params=request.model_params,
                    task_type=None,
                    mlflow_experiment_name = request.mlflow_experiment_name
                    )

                # Получаем данные для обучения: 
                # Если пришел путь до файла, то скачаем из minio, если пришли данные в запросе, то сохраним датасет в minio
                
                if request.dataset_path is not None:
                    file_path = os.path.join('./download_data', request.dataset_path)
                    download_dataset_from_minio(minio_client, os.getenv("MINIO_BUCKET_NAME", "mlopsbucket"), filename, file_path)

                    df = pd.read_csv(file_path)
                    X = df.drop(columns='target')
                    y = df['target']

                    # Удаление файла из буферной папки после использования
                    os.remove(file_path)
                    logger.info(f'Buffer file {file_path} deleted')
                else:
                    feature_names = request.feature_names if len(request.feature_names) else None
                    X = pd.DataFrame(data=request.features, columns=feature_names)
                    y = request.target
                    df = X.copy()
                    df['target'] = y

                    filename = model.model_id + '.csv'
                    file_path = os.path.join('./upload_data', filename)
                    df.to_csv(file_path, index=False)
                    upload_dataset_to_minio_and_dvc_track(minio_client, os.getenv("MINIO_BUCKET_NAME", "mlopsbucket"), filename, file_path)

                model.retrain(X, y, 
                    cv=request.cv, 
                    optimize_hyperparameters_flag=request.optimize_hyperparameters_flag, 
                    optimize_hyperparameters_params=request.optimize_hyperparameters_params)

                return {"model_id": model.model_id, "status": "retrained"}

    raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")

@app.post("/evaluate", response_model=Dict[str, Any])
def evaluate_model(request: EvaluateRequest):
    """
    Оцениваем по основным метрикам уже обученную модель на отправленных данных.
    """
    model_existing_flag, model_type = check_model_status(request.model_id)
    if not model_existing_flag:
        raise HTTPException(status_code=404, detail=f"Model with id {model_id} not found")

    for file, classes in available_models.items():
        for cls in classes:
            if cls.__name__ == model_type:
                model = cls(model_id=request.model_id, task_type=None)

                # Получаем данные для предикта: 
                # Если пришел путь до файла, то скачаем из minio
                
                if request.dataset_path is not None:
                    file_path = os.path.join('./download_data', request.dataset_path)
                    download_dataset_from_minio(minio_client, MINIO_BUCKET_NAME, request.dataset_path , file_path)

                    df = pd.read_csv(local_dataset_path)
                    X = df.drop(columns='target')
                    y = df['target']

                    # Удаление файла из буферной папки после использования
                    os.remove(file_path)
                    logger.info(f'Buffer file {file_path} deleted')
                else:
                    feature_names = request.feature_names if len(request.feature_names) else None
                    X = pd.DataFrame(data=request.features, columns=feature_names)
                    y = request.target

                result_metrics = model.evaluate(X, y)

                return result_metrics

    raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")

@app.delete("/delete/{model_id}", response_model=Dict[str, Any])
def delete_model(model_id: str):
    """
    Удаляет уже обученную модель.
    """
    model_existing_flag, model_type = check_model_status(model_id)
    if not model_existing_flag:
        raise HTTPException(status_code=404, detail=f"Model with id {model_id} not found")

    for file, classes in available_models.items():
        for cls in classes:
            if cls.__name__ == model_type:
                model = cls(model_id=model_id)
                model.delete()

    return {"model_id": model_id, "status": "deleted"}