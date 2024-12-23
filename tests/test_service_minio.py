import pytest
import requests
import json
from ml_models.create_datasets import create_dataset
import pandas as pd
import numpy as np

# URL вашего сервиса
BASE_URL = "http://127.0.0.1:8000"

# Функция для создания данных
def generate_data(task_type, n_numeric, n_categorical, n_samples=100, n_classes=2):
    df = create_dataset(task_type, n_numeric, n_categorical, n_samples,n_classes=n_classes)
    features = df.drop(columns=['target']).values.tolist()
    target = df['target'].values.tolist()
    feature_names = df.drop(columns=['target']).columns.tolist()
    return features, target, feature_names

# Тест для получения списка моделей
def test_get_models():
    response = requests.get(f"{BASE_URL}/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

# Тест для получения статуса сервиса
def test_get_status():
    response = requests.get(f"{BASE_URL}/status")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert 'status' in data

# Тест для обучения модели
@pytest.mark.parametrize("task_type, model_type, n_classes", [
    ("regression", "LinearModel", 2),
    ("regression", "CatBoostModel", 2),
    ("regression", "TreeModel", 2),
    ("binary_classification", "LinearModel", 2),
    ("binary_classification", "CatBoostModel", 2),
    ("binary_classification", "TreeModel", 2),
    ("multiclass_classification", "LinearModel", 3),
    ("multiclass_classification", "CatBoostModel", 4),
    ("multiclass_classification", "TreeModel", 3)
])
def test_train_model(task_type, model_type, n_classes):
    features, target, feature_names = generate_data(task_type, 10, 2,n_classes=n_classes)
    target = target if np.random.rand() > 0.3 or 'reg' in task_type else list(map(str, target))
    data = {
        "model_type": model_type,
        "task_type": task_type,
        "feature_names": feature_names,
        "features": features,
        "target": target,
        "cv": 5,
        "optimize_hyperparameters_flag": False,
        "optimize_hyperparameters_params": {}
    }
    response = requests.post(f"{BASE_URL}/train", json=data)
    assert response.status_code == 200
    data = response.json()
    assert 'model_id' in data

# Тест для получения информации о моделях
def test_get_models_info():
    response = requests.get(f"{BASE_URL}/models_info")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

# Тест для оценки модели
@pytest.mark.parametrize("task_type, model_type, n_classes", [
    ("regression", "LinearModel", 2),
    ("regression", "CatBoostModel", 2),
    ("regression", "TreeModel", 2),
    ("binary_classification", "LinearModel", 2),
    ("binary_classification", "CatBoostModel", 2),
    ("binary_classification", "TreeModel", 2),
    ("multiclass_classification", "LinearModel", 3),
    ("multiclass_classification", "CatBoostModel", 4),
    ("multiclass_classification", "TreeModel", 3)
])
def test_evaluate_model(task_type, model_type, n_classes):
    features, target, feature_names = generate_data(task_type, 10, 2,n_classes=n_classes)
    target = target if np.random.rand() > 0.3 or 'reg' in task_type else list(map(str, target))
    data = {
        "model_type": model_type,
        "task_type": task_type,
        "feature_names": feature_names,
        "features": features,
        "target": target,
        "cv": 5,
        "optimize_hyperparameters_flag": False,
        "optimize_hyperparameters_params": {}
    }
    response = requests.post(f"{BASE_URL}/train", json=data)
    model_id = response.json()['model_id']
    
    data = {
        "model_id": model_id,
        "feature_names": feature_names,
        "features": features,
        "target": target
    }
    response = requests.post(f"{BASE_URL}/evaluate", json=data)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)

# Тест для предсказания модели
@pytest.mark.parametrize("task_type, model_type, n_classes", [
    ("regression", "LinearModel", 2),
    ("regression", "CatBoostModel", 2),
    ("regression", "TreeModel", 2),
    ("binary_classification", "LinearModel", 2),
    ("binary_classification", "CatBoostModel", 2),
    ("binary_classification", "TreeModel", 2),
    ("multiclass_classification", "LinearModel", 3),
    ("multiclass_classification", "CatBoostModel", 4),
    ("multiclass_classification", "TreeModel", 3)
])
def test_predict_model(task_type, model_type, n_classes):
    features, target, feature_names = generate_data(task_type, 10, 2,n_classes=n_classes)
    target = target if np.random.rand() > 0.3 or 'reg' in task_type else list(map(str, target))
    data = {
        "model_type": model_type,
        "task_type": task_type,
        "feature_names": feature_names,
        "features": features,
        "target": target,
        "cv": 5,
        "optimize_hyperparameters_flag": False,
        "optimize_hyperparameters_params": {}
    }
    response = requests.post(f"{BASE_URL}/train", json=data)
    model_id = response.json()['model_id']
    
    data = {
        "model_id": model_id,
        "prediction_type": "predict",
        "feature_names": feature_names,
        "features": features
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

# Тест для переобучения модели
@pytest.mark.parametrize("task_type, model_type, n_classes", [
    ("regression", "LinearModel", 2),
    ("regression", "CatBoostModel", 2),
    ("regression", "TreeModel", 2),
    ("binary_classification", "LinearModel", 2),
    ("binary_classification", "CatBoostModel", 2),
    ("binary_classification", "TreeModel", 2),
    ("multiclass_classification", "LinearModel", 3),
    ("multiclass_classification", "CatBoostModel", 4),
    ("multiclass_classification", "TreeModel", 3)
])
def test_retrain_model(task_type, model_type, n_classes):
    features, target, feature_names = generate_data(task_type, 10, 2,n_classes=n_classes)
    target = target if np.random.rand() > 0.3 or 'reg' in task_type else list(map(str, target))
    data = {
        "model_type": model_type,
        "task_type": task_type,
        "feature_names": feature_names,
        "features": features,
        "target": target,
        "cv": 5,
        "optimize_hyperparameters_flag": False,
        "optimize_hyperparameters_params": {}
    }
    response = requests.post(f"{BASE_URL}/train", json=data)
    model_id = response.json()['model_id']
    
    data = {
        "model_id": model_id,
        "task_type": task_type,
        "feature_names": feature_names,
        "features": features,
        "target": target,
        "cv": 5,
        "optimize_hyperparameters_flag": False,
        "optimize_hyperparameters_params": {}
    }
    response = requests.post(f"{BASE_URL}/retrain", json=data)
    assert response.status_code == 200
    data = response.json()
    assert 'model_id' in data

# Тест для удаления модели
@pytest.mark.parametrize("task_type, model_type, n_classes", [
    ("regression", "LinearModel", 2),
    ("regression", "CatBoostModel", 2),
    ("regression", "TreeModel", 2),
    ("binary_classification", "LinearModel", 2),
    ("binary_classification", "CatBoostModel", 2),
    ("binary_classification", "TreeModel", 2),
    ("multiclass_classification", "LinearModel", 3),
    ("multiclass_classification", "CatBoostModel", 4),
    ("multiclass_classification", "TreeModel", 3)
])
def test_delete_model(task_type, model_type, n_classes):
    features, target, feature_names = generate_data(task_type, 10, 2,n_classes=n_classes)
    target = target if np.random.rand() > 0.3 or 'reg' in task_type else list(map(str, target))
    data = {
        "model_type": model_type,
        "task_type": task_type,
        "feature_names": feature_names,
        "features": features,
        "target": target,
        "cv": 5,
        "optimize_hyperparameters_flag": False,
        "optimize_hyperparameters_params": {}
    }
    response = requests.post(f"{BASE_URL}/train", json=data)
    model_id = response.json()['model_id']
    
    response = requests.delete(f"{BASE_URL}/delete/{model_id}")
    assert response.status_code == 200
    data = response.json()
    assert data['model_id'] == model_id
    assert data['status'] == 'deleted'