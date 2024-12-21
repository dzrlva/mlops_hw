# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.utils import find_model_files, import_module_from_file, get_model_classes
from ml_models.base_model import BaseModel as BaseMLModel
from typing import List, Dict, Any
import os
import joblib
import pandas as pd

app = FastAPI(title="ML Model Service")

@app.get("/models", response_model=List[str])
def get_available_models():
    """
    Возвращает список доступных для обучения классов моделей.
    """
    # Загрузка классов доступных моделей
    models_dir = 'ml_models'
    model_files = find_model_files(models_dir)
    available_models = {}

    for file in model_files:
        file_path = os.path.join(models_dir, file)
        module = import_module_from_file(file_path)
        model_classes = get_model_classes(module)
        available_models[file] = model_classes[0]
        available_model_names = []

    for file, classes in available_models.items():
        available_model_names.extend([cls.__name__ for cls in classes])
    
    return available_model_names

@app.post("/train/{model_name}", response_model=Dict[str, Any])
def train_model(model_name: str, params: Dict[str, Any]):
    """
    Обучает ML-модель с возможностью настройки гиперпараметров.
    """
    for file, classes in available_models.items():
        for cls in classes:
            if cls.__name__ == model_name:
                model = cls(**params)
                model.train(X, y)  # Здесь нужно заменить X и y на реальные данные
                return {"model_id": model.model_id, "status": "trained"}
    raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

@app.post("/predict/{model_id}", response_model=List[float])
def predict(model_id: str, X: List[List[float]]):
    """
    Возвращает предсказание конкретной модели.
    """
    if model_id not in trained_models:
        # Проверка в журнале моделей
        model_info = log_df[log_df['model_id'] == model_id]
        if model_info.empty:
            raise HTTPException(status_code=404, detail=f"Model with id {model_id} not found")
        model_path = model_info['model_path'].values[0]
        model_params = eval(model_info['model_params'].values[0])
        task_type = model_info['task_type'].values[0]
        for file, classes in available_models.items():
            for cls in classes:
                if cls.__name__ == model_info['model_params'].values[0].split("'")[1]:
                    model = cls(model_id=model_id, model_params=model_params, task_type=task_type)
                    model.load()
                    trained_models[model_id] = model
                    break
        if model_id not in trained_models:
            raise HTTPException(status_code=404, detail=f"Model with id {model_id} not found")
    model = trained_models[model_id]
    # Предполагается, что у модели есть метод predict
    return model.predict(X)

@app.post("/retrain/{model_id}", response_model=Dict[str, Any])
def retrain_model(model_id: str, params: Dict[str, Any]):
    """
    Обучает заново уже обученную модель.
    """
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail=f"Model with id {model_id} not found")
    model = trained_models[model_id]
    # Предполагается, что у модели есть метод train
    model.train(X, y)  # Здесь нужно заменить X и y на реальные данные
    return {"model_id": model.model_id, "status": "retrained"}

@app.delete("/delete/{model_id}", response_model=Dict[str, Any])
def delete_model(model_id: str):
    """
    Удаляет уже обученную модель.
    """
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail=f"Model with id {model_id} not found")
    model = trained_models[model_id]
    model.delete()
    del trained_models[model_id]
    return {"model_id": model_id, "status": "deleted"}

@app.get("/status", response_model=Dict[str, str])
def get_status():
    """
    Возвращает статус сервиса.
    """
    return {"status": "OK", "models_count": str(len(trained_models))}