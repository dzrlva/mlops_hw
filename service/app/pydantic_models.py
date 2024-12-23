from pydantic import BaseModel
from typing import Dict, Any, List

class TrainRequest(BaseModel):
    """
    Модель запроса для инициализации и обучения модели.
    :param model_type: Тип модели из списка доступных.
    :param mlflow_experiment_name: Эксперимент в mlflow
    :param model_description: Описание модели
    :param model_params: Параметры модели.
    :param task_type: Тип задачи (regression, binary_classification, multiclass_classification)
    :param dataset_path: Путь до датасета
    :param feature_names: Список признаков соотвествующий данным
    :param features: Данные для обучения.
    :param target: Целевые значения для обучения.
    :param cv: Количество фолдов для кросс-валидации.
    :param optimize_hyperparameters_flag: Флаг для автоматического подбора гиперпараметров.
    :param optimize_hyperparameters_params: Параметры для подбора гиперпараметров под GridSearchCV.
    """
    model_type: str
    mlflow_experiment_name: str = 'mlflow_model'
    model_description: str = None
    model_params: Dict[str, Any] = {}
    task_type: str = 'regression'
    dataset_path: str = None
    feature_names: List[str] = []
    features: List[List[Any]] = [[]]
    target: List[Any] = []
    cv: int = 5
    optimize_hyperparameters_flag: bool = False
    optimize_hyperparameters_params: Dict[str, Any] = {}

class PredictRequest(BaseModel):
    """
    Модель запроса для предсказания.
    :param model_id: Уникальный идентификатор модели.
    :param prediction_type: Тип предсказания
    :param dataset_path: Путь до датасета    
    :param feature_names: Список признаков соотвествующий данным
    :param features: Данные для предсказания.
    """
    model_id: str
    prediction_type: str
    dataset_path: str = None
    feature_names: List[str] = []
    features: List[List[Any]] = [[]]

class RetrainRequest(BaseModel):
    """
    Модель запроса для переобучения обучения модели.
    :param model_id: Уникальный идентификатор модели.
    :param mlflow_experiment_name: Эксперимент в mlflow
    :param model_description: Описание модели
    :param model_params: Параметры модели.
    :param dataset_path: Путь до датасета
    :param feature_names: Список признаков соотвествующий данным
    :param features: Данные для обучения.
    :param target: Целевые значения для обучения.
    :param cv: Количество фолдов для кросс-валидации.
    :param optimize_hyperparameters_flag: Флаг для автоматического подбора гиперпараметров.
    :param optimize_hyperparameters_params: Параметры для подбора гиперпараметров под GridSearchCV.
    """
    model_id: str
    mlflow_experiment_name: str = 'mlflow_model'
    model_description: str = None
    model_params: Dict[str, Any] = {}
    task_type: str = 'regression'
    dataset_path: str = None
    feature_names: List[str] = []
    features: List[List[Any]] = [[]]
    target: List[Any] = []
    cv: int = 5
    optimize_hyperparameters_flag: bool = False
    optimize_hyperparameters_params: Dict[str, Any] = {}

class EvaluateRequest(BaseModel):
    """
    Модель запроса для оценки модели на данных.
    :param model_id: Уникальный идентификатор модели.
    :param dataset_path: Путь до датасета
    :param feature_names: Список признаков соотвествующий данным
    :param features: Данные для оценки.
    :param target: Целевые значения для оценки.
    """
    model_id: str
    dataset_path: str = None
    feature_names: List[str] = []
    features: List[List[Any]] = [[]]
    target: List[Any] = []