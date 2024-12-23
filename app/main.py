from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import joblib
import pandas as pd
from datetime import datetime

from minio import Minio
import dvc.api
import mlflow
from mlflow.models import Model
from mlflow.sklearn import save_model, load_model
from botocore.exceptions import NoCredentialsError, ClientError

from ml_models.catboost_model import CatBoostModel
from ml_models.linear_model import LinearModel
from ml_models.tree_model import TreeModel
from app.utils import find_model_files, import_module_from_file, get_model_classes, check_model_status
from app.pydantic_models import PredictRequest, RetrainRequest, EvaluateRequest, TrainRequest

import logging

app = FastAPI(title="ML Model Service")

# Настройка логгирования
logging.basicConfig(filename='./ml_service_logs', filemode='a', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('./ml_service_logs')

# Настройка Minio
minio_client = Minio(
    os.getenv("MINIO_ENDPOINT", "play.min.io"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    secure=False  # Установите True, если используете HTTPS
)

# Настройка MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

# Загрузка классов доступных моделей
models_dir = './ml_models'
model_files = find_model_files(models_dir)
available_models = {}

for file in model_files:
    file_path = os.path.join(models_dir, file)
    module = import_module_from_file(file_path)
    model_classes = get_model_classes(module)
    available_models[file] = model_classes

# Настройки Minio
MINIO_BUCKET_NAME = 'mlops_mybucket'

# Инициализация Minio клиента
minio_client = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY
)

# Создание бакета в Minio
def create_minio_bucket(bucket_name):
    try:
        minio_client.create_bucket(Bucket=bucket_name)
        logger.info(f'Bucket {bucket_name} created successfully.')
    except ClientError as e:
        logger.error(f'Error creating bucket {bucket_name}: {e}')
        raise HTTPException(status_code=500, detail=str(e))

# Инициализация DVC и настройка удаленного хранилища
def setup_dvc():
    try:
        # Инициализация DVC
        dvc.api.init()
        # Настройка удаленного хранилища DVC
        dvc.api.remote_add(name='myremote', url=f's3://{MINIO_BUCKET_NAME}')
        dvc.api.remote_modify(name='myremote', option='endpointurl', value=os.getenv("MINIO_ENDPOINT", "play.min.io"),)
        dvc.api.remote_modify(name='myremote', option='access_key_id', value=os.getenv("MINIO_ACCESS_KEY", "minioadmin"))
        dvc.api.remote_modify(name='myremote', option='secret_access_key', value=os.getenv("MINIO_SECRET_KEY", "minioadmin"))
        dvc.api.remote_default(name='myremote')

        logger.info('DVC initialized and remote storage configured successfully.')
    except Exception as e:
        logger.error(f'Error setting up DVC: {e}')
        raise HTTPException(status_code=500, detail=str(e))

# Создание бакета и настройка DVC при запуске приложения
@app.on_event("startup")
async def startup_event():
    create_minio_bucket(MINIO_BUCKET_NAME)
    setup_dvc()

@app.get("/models", response_model=List[str])
def get_available_models():
    """
    Возвращает список доступных для обучения классов моделей.
    """
    available_model_names = []
    for file, classes in available_models.items():
        available_model_names.extend([cls.__name__ for cls in classes])
    
    return available_model_names

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

@app.post("/upload-dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        # Сохранение файла локально
        file_path = os.path.join('./data', file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Загрузка файла в Minio
        minio_client.fput_object("datasets", file.filename, file_path)
        logger.info(f'Dataset uploaded to Minio: {file_path} to {file.filename}')

        # Отслеживание файла с помощью DVC
        dvc.api.add(file_path)
        dvc.api.commit('Add dataset')
        logger.info(f'Dataset tracked with DVC: {file_path}')

        return {"filename": file.filename, "message": "Dataset uploaded and tracked successfully"}
    except Exception as e:
        logger.error(f'Error uploading and tracking dataset: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-dataset/{filename}")
async def download_dataset(filename: str):
    try:
        # Путь к файлу локально
        file_path = os.path.join('./data', filename)

        # Скачивание файла из Minio
        minio_client.fget_object("datasets", filename, file_path)
        logger.info(f'Dataset downloaded from Minio: {filename} to {file_path}')

        return {"filename": filename, "message": "Dataset downloaded and tracked successfully"}
    except Exception as e:
        logger.error(f'Error downloading and tracking dataset: {e}')
        raise HTTPException(status_code=500, detail=str(e))

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

                if request.dataset_path is not None:
                    local_dataset_path = os.path.join('./data', request.dataset_path)
                    model.download_dataset_from_minio(minio_client, MINIO_BUCKET_NAME, request.dataset_path , local_dataset_path)

                    df = pd.read_csv(local_dataset_path)
                    X = df.drop(columns='target')
                    y = df['target']
                else:
                    feature_names = request.feature_names if len(request.feature_names) else None
                    X = pd.DataFrame(data=request.features, columns=feature_names)
                    y = request.target
                
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

    if request.dataset_path is not None:
        local_dataset_path = os.path.join('./data', request.dataset_path)
        model.download_dataset_from_minio(minio_client, MINIO_BUCKET_NAME, request.dataset_path , local_dataset_path)

        df = pd.read_csv(local_dataset_path)
        if 'target' in df.columns:
            X = df.drop(columns='target')
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
                
                if request.dataset_path is not None:
                    local_dataset_path = os.path.join('./data', request.dataset_path)
                    model.download_dataset_from_minio(minio_client, MINIO_BUCKET_NAME, request.dataset_path , local_dataset_path)

                    df = pd.read_csv(local_dataset_path)
                    X = df.drop(columns='target')
                    y = df['target']
                else:
                    feature_names = request.feature_names if len(request.feature_names) else None
                    X = pd.DataFrame(data=request.features, columns=feature_names)
                    y = request.target

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
                
                if request.dataset_path is not None:
                    local_dataset_path = os.path.join('./data', request.dataset_path)
                    model.download_dataset_from_minio(minio_client, MINIO_BUCKET_NAME, request.dataset_path , local_dataset_path)

                    df = pd.read_csv(local_dataset_path)
                    X = df.drop(columns='target')
                    y = df['target']
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