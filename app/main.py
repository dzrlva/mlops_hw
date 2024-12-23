from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import joblib
import pandas as pd
from datetime import datetime

from ml_models.catboost_model import CatBoostModel
from ml_models.linear_model import LinearModel
from ml_models.tree_model import TreeModel
from app.utils import find_model_files, import_module_from_file, get_model_classes, check_model_status
from app.pydantic_models import PredictRequest, RetrainRequest, EvaluateRequest, TrainRequest

import logging

app = FastAPI(title="ML Model Service")

# Загрузка классов доступных моделей
models_dir = './ml_models'
model_files = find_model_files(models_dir)
available_models = {}

for file in model_files:
    file_path = os.path.join(models_dir, file)
    module = import_module_from_file(file_path)
    model_classes = get_model_classes(module)
    available_models[file] = model_classes

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
                    task_type=request.task_type)

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
                    task_type=None
                    )
                
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