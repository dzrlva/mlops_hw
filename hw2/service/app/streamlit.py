import os
import streamlit as st
import requests
from app.pydantic_models import TrainRequest, RetrainRequest, EvaluateRequest, PredictRequest

# URL вашего REST сервиса
API_URL = os.getenv("FASTAPI_ENDPOINT", "http://fastapi:8000")

# Функция для получения статуса сервиса
def get_status():
    response = requests.get(f"{API_URL}/status")
    return response.json()

# Функция для получения списка доступных моделей
def get_available_models():
    response = requests.get(f"{API_URL}/models")
    return response.json()

# Функция для получения списка доступных датасетов
def get_available_datasets():
    response = requests.get(f"{API_URL}/datasets_info")
    return response.json()

# Функция для получения списка моделей из журнала
def get_journal_models():
    response = requests.get(f"{API_URL}/models_info")
    return response.json()

# Функция для обучения новой модели
def train_model(request: TrainRequest):
    response = requests.post(f"{API_URL}/train", json=request.dict())
    return response.json()

# Функция для переобучения существующей модели
def retrain_model(request: RetrainRequest):
    response = requests.post(f"{API_URL}/retrain", json=request.dict())
    return response.json()

# Функция для оценки модели
def evaluate_model(request: EvaluateRequest):
    response = requests.post(f"{API_URL}/evaluate", json=request.dict())
    return response.json()

# Функция для предсказания с помощью модели
def predict_model(request: PredictRequest):
    response = requests.post(f"{API_URL}/predict", json=request.dict())
    return response.json()

# Функция для удаления модели
def delete_model(model_id: str):
    response = requests.delete(f"{API_URL}/delete/{model_id}")
    return response.json()

# Основное меню дэшборда
st.title("Машинное обучение с использованием REST сервиса")

menu = st.sidebar.selectbox("Выберите действие", ["Загрузка датасета", "Скачивание датасета", "Доступные датасеты",
    "Список моделей", "Обучение новой модели", "Переобучение модели", "Оценка модели", "Предсказание", "Удаление модели", "Информация о моделях", "Статус"])

if menu == "Загрузка датасета":
    st.header("Загрузка датасета")
    uploaded_file = st.file_uploader("Выберите файл (в файле обязательно должна быть переменная target)")
    if uploaded_file is not None:
        # Загрузка файла через REST сервис
        files = {'file': uploaded_file}
        response = requests.post(f"{API_URL}/upload_dataset/", files=files)
        st.write(response.json())

elif menu == "Доступные датасеты":
    st.header("Доступные датасеты")
    datasets = get_available_datasets()
    st.write(datasets)

elif menu == "Скачивание датасета":
    st.header("Скачивание датасета")
    object_name = st.text_input("Название датасета в Minio")
    # Получение датасета
    if st.button("Скачать"):
        try:
            response = requests.get(f"{API_URL}/download_dataset/{object_name}")
            if response.status_code == 200:
                st.download_button(
                    label="Скачать датасет",
                    data=response.content,
                    file_name=object_name,
                    mime="text/csv"
                )
            else:
                st.error(f"Ошибка при скачивании датасета: {response.json()}")
        except Exception as e:
            st.error(f'Error downloading dataset: {e}')

elif menu == "Список моделей":
    st.header("Список доступных моделей")
    models = get_available_models()
    st.write(models)

elif menu == "Обучение новой модели":
    st.header("Обучение новой модели")
    available_model_names = get_available_models()
    model_type = st.selectbox("Тип модели", available_model_names)
    mlflow_experiment_name = st.text_input("Имя эксперимента в mlflow")
    model_description = st.text_input("Описание модели")
    model_params = st.text_area("Параметры модели (JSON)")
    task_type = st.selectbox("Тип задачи", ["regression", "binary_classification", "multiclass_classification"])
    dataset_path = st.text_input("Путь до датасета")
    feature_names = st.text_area("Список признаков (JSON)")
    features = st.text_area("Данные для обучения (JSON)")
    target = st.text_area("Целевые значения для обучения (JSON)")
    cv = st.number_input("Количество фолдов для кросс-валидации", min_value=1, value=5)
    optimize_hyperparameters_flag = st.checkbox("Автоматический подбор гиперпараметров")
    optimize_hyperparameters_params = st.text_area("Параметры для подбора гиперпараметров (JSON)")

    if st.button("Обучить модель"):
        request = TrainRequest(
            model_type=model_type,
            mlflow_experiment_name=mlflow_experiment_name if mlflow_experiment_name else 'mlflow_model',
            model_description=model_description,
            model_params=eval(model_params) if model_params else {},
            task_type=task_type,
            dataset_path=dataset_path if dataset_path else None,
            feature_names=eval(feature_names) if feature_names else [],
            features=eval(features) if features else [[]],
            target=eval(target) if target else [],
            cv=cv,
            optimize_hyperparameters_flag=optimize_hyperparameters_flag,
            optimize_hyperparameters_params=eval(optimize_hyperparameters_params) if optimize_hyperparameters_params else {}
        )
        result = train_model(request)
        st.write(result)

elif menu == "Переобучение модели":
    st.header("Переобучение существующей модели")
    model_id = st.text_input("ID модели")
    mlflow_experiment_name = st.text_input("Имя эксперимента в mlflow")
    model_description = st.text_input("Описание модели")
    model_params = st.text_area("Параметры модели (JSON)")
    dataset_path = st.text_input("Путь до датасета")
    feature_names = st.text_area("Список признаков (JSON)")
    features = st.text_area("Данные для обучения (JSON)")
    target = st.text_area("Целевые значения для обучения (JSON)")
    cv = st.number_input("Количество фолдов для кросс-валидации", min_value=2, value=5)
    optimize_hyperparameters_flag = st.checkbox("Автоматический подбор гиперпараметров")
    optimize_hyperparameters_params = st.text_area("Параметры для подбора гиперпараметров (JSON)")

    if st.button("Переобучить модель"):
        request = RetrainRequest(
            model_id=model_id,
            mlflow_experiment_name=mlflow_experiment_name if mlflow_experiment_name else 'mlflow_model',
            model_description=model_description,
            model_params=eval(model_params) if model_params else {},
            dataset_path=dataset_path if dataset_path else None,
            feature_names=eval(feature_names) if feature_names else [],
            features=eval(features) if features else [[]],
            target=eval(target) if target else [],
            cv=cv,
            optimize_hyperparameters_flag=optimize_hyperparameters_flag,
            optimize_hyperparameters_params=eval(optimize_hyperparameters_params) if optimize_hyperparameters_params else {}
        )
        result = retrain_model(request)
        st.write(result)

elif menu == "Оценка модели":
    st.header("Оценка модели")
    model_id = st.text_input("ID модели")
    dataset_path = st.text_input("Путь до датасета")
    feature_names = st.text_area("Список признаков (JSON)")
    features = st.text_area("Данные для оценки (JSON)")
    target = st.text_area("Целевые значения для оценки (JSON)")

    if st.button("Оценить модель"):
        request = EvaluateRequest(
            model_id=model_id,
            dataset_path=dataset_path if dataset_path else None,
            feature_names=eval(feature_names) if feature_names else [],
            features=eval(features) if features else [[]],
            target=eval(target) if target else [],
        )
        result = evaluate_model(request)
        st.write(result)

elif menu == "Предсказание":
    st.header("Предсказание с помощью модели")
    model_id = st.text_input("ID модели")
    prediction_type = st.selectbox("Тип предсказания", ["predict", "predict_proba"])
    dataset_path = st.text_input("Путь до датасета")
    feature_names = st.text_area("Список признаков (JSON)")
    features = st.text_area("Данные для предсказания (JSON)")

    if st.button("Предсказать"):
        request = PredictRequest(
            model_id=model_id,
            prediction_type=prediction_type,
            dataset_path=dataset_path if dataset_path else None,
            feature_names=eval(feature_names) if feature_names else [],
            features=eval(features) if features else [[]],
        )
        result = predict_model(request)
        st.write(result)

elif menu == "Удаление модели":
    st.header("Удаление модели")
    model_id = st.text_input("ID модели")

    if st.button("Удалить модель"):
        result = delete_model(model_id)
        st.write(result)

elif menu == "Информация о моделях":
    st.header("Информация о моделях")

    result = get_journal_models()
    st.write(result)

elif menu == "Статус":
    st.header("Статус сервиса")

    result = get_status()
    st.write(result)
    
