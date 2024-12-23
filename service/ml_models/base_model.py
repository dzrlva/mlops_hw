import os
import joblib
import pandas as pd
import logging
from uuid import uuid4
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import (
    median_absolute_error, max_error, mean_absolute_percentage_error, mean_absolute_error,
    mean_squared_error, mean_squared_log_error, r2_score, explained_variance_score,
    mean_gamma_deviance, mean_poisson_deviance, accuracy_score, precision_score,
    recall_score, fbeta_score, roc_auc_score, average_precision_score
)
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

import mlflow
from mlflow.models import Model
from mlflow.sklearn import save_model, load_model

from ml_models.task_metrics import reg_model_metric_performance, binary_model_metric_performance, multiclass_model_metric_performance

# Настройка логгирования
logging.basicConfig(filename='./ml_service_logs', filemode='a', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('./ml_service_logs')

# Настройка MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

class BaseModel(ABC):
    """
    Базовый класс для модели машинного обучения.
    """
    def __init__(self, model_id=None, model_description=None, model_params=None, task_type='regression',mlflow_experiment_name='mlflow_model'):
        """
        Инициализация модели.
        :param model_id: Уникальный идентификатор модели.
        :param model_description: Описание модели
        :param model_params: Параметры модели.
        :param task_type: Тип задачи (regression, binary_classification, multiclass_classification).
        :param mlflow_experiment_name: Название эксперимента в mlflow
        """
        self.model_id = model_id if model_id else str(uuid4())
        self.model_params = model_params if model_params else {}
        self.model = None
        self.status = 'initialized'
        self.model_path = os.path.join('models', f'{self.model_id}.joblib')
        self.log_path = 'models_log.csv'
        self.task_type = task_type
        self.model_description = model_description
        self.le_path = os.path.join('label_encoders', f'{self.model_id}_le.joblib') # LabelEncoder при необходимости
        self.mlflow_experiment_name = mlflow_experiment_name
        # Создаем директорию для моделей, если она не существует
        if not os.path.exists('models'):
            os.makedirs('models')
        if not os.path.exists('label_encoders'):
            os.makedirs('label_encoders')
        # Создаем журнал моделей, если он не существует
        if not os.path.exists(self.log_path):
            log_df = pd.DataFrame(columns=['model_id', 'model_type', 'model_description', 'model_params', 'mlflow_experiment_name',
                'status', 'status_time', 'model_path', 'task_type', 'cv_metrics'])
            log_df.to_csv(self.log_path, index=False)
        if model_id is None:
            # Записываем информацию о модели в журнал
            self._update_log()
            logger.info(f'Model {self.model_id} initialized with params {self.model_params} for task {self.task_type}')
        else:
            # Загружаем логи по данной модели
            model_id_logs = pd.read_csv(self.log_path, parse_dates=['status_time'])
            model_id_logs = model_id_logs.loc[model_id_logs.model_id == self.model_id].sort_values('status_time')
            # Восстановим из логов не заполненные параметры, такие как task_type, которые понадобятся            
            if self.task_type is None:
                self.task_type = model_id_logs.iloc[-1, 8]
            if self.model_params is None:
                self.model_params = eval(model_id_logs.iloc[-1, 3])
            if self.model_description is None:
                self.model_description = model_id_logs.iloc[-1, 2]
            if self.mlflow_experiment_name is None:
                self.mlflow_experiment_name = model_id_logs.iloc[-1, 4]

        self.start_mlflow_experiment()

    def _update_log(self, cv_metrics=None):
        """
        Обновление журнала моделей.
        :param cv_metrics: Метрики кросс-валидации.
        """
        # создаем новую запись
        new_log = pd.DataFrame({
            'model_id': [self.model_id],
            'model_type': [self.__class__.__name__],
            'model_description': [self.model_description],
            'model_params': [str(self.model_params)],
            'status': [self.status],
            'status_time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'model_path': [self.model_path],
            'task_type': [self.task_type],
            'cv_metrics': [str(cv_metrics) if cv_metrics else None]
        })
        with open(self.log_path, 'a') as f:
            new_log.to_csv(f, index=False, header=False)
            
    @abstractmethod
    def prepare_features(self, X, y):
        """
        Подготовка признаков.
        :param X: Матрица признаков.
        :param y: Вектор целевых значений.
        :return: Подготовленные признаки.
        """
        pass

    @abstractmethod
    def fit(self, X, y):
        """
        Обучение модели.
        :param X: Матрица признаков.
        :param y: Вектор целевых значений.
        """
        pass

    @abstractmethod
    def optimize_hyperparameters(self, X, y, optimize_hyperparameters_params):
        """
        Подбор гиперпараметров модели.
        :param X: Матрица признаков.
        :param y: Вектор целевых значений.
        :param optimize_hyperparameters_params: Параметры для подбора гиперпараметров под GridSearchCV.
        :return: Оптимальные гиперпараметры.
        """
        pass

    def prepare_taget(self, y):
        """
        Предобработка признаков.
        :param y: Целевые значения.
        :return: Подготовленные целевые значения.
        """
        if self.task_type == 'regression':
            if all(isinstance(x, (int, float)) for x in y):
                return np.array(y, dtype=float)
            else:
                raise Exception("Все значения целевой переменной должны быть числами для задачи регрессии.")
        elif 'classification' in self.task_type:
            if not all(isinstance(x, (int, float)) for x in y):
                # Есть строковые данные, нужен LabelEncoder
                target = np.array(y, dtype=str)

                if not os.path.exists(self.le_path):
                    label_encoder = LabelEncoder()
                    label_encoder.fit(target)
                    # Сохранение пайплайна
                    joblib.dump(label_encoder, self.le_path)
                    logger.info(f'LabelEncoder {self.model_id} saved to {self.le_path}')
                else:
                    label_encoder = joblib.load(self.le_path)
                    logger.info(f'LabelEncoder {self.model_id} loaded from {self.le_path}')

                target_encoded = label_encoder.transform(target)
                return target_encoded
            else:
                # Если все значения числовые, преобразуем их к целым числам
                target = np.array(y, dtype=int)
                return target
        else:
            logger.error(f'Unknown task type {self.task_type}')
            raise Exception('Unknown task type')

    def train(self, X, y, cv=5, optimize_hyperparameters_flag=False, optimize_hyperparameters_params=None):
        """
        Обучение модели с оценкой метрик на кросс-валидации и подбором гиперпараметров.
        :param X: Матрица признаков.
        :param y: Вектор целевых значений.
        :param cv: Количество фолдов для кросс-валидации.
        :param optimize_hyperparameters_flag: Флаг для автоматического подбора гиперпараметров.
        :param optimize_hyperparameters_params: Параметры для подбора гиперпараметров под GridSearchCV.
        """
        X_prepared = self.prepare_features(X, y)
        y_prepared = self.prepare_taget(y)

        # Подбор гиперпараметров, если флаг установлен
        if optimize_hyperparameters_flag:
            self.model_params = self.optimize_hyperparameters(X_prepared, y_prepared, optimize_hyperparameters_params)
            logger.info(f'Model {self.model_id} optimized with params {self.model_params}')
        
        self.status = 'fitting' if self.status != 'retrained' else 'refitting'
        self._update_log()
        self.fit(X_prepared, y_prepared)
        
        # Оценка метрик на кросс-валидации
        if self.task_type == 'regression':
            cv_scores = cross_val_score(self.model, X_prepared, y_prepared, cv=cv, scoring='neg_mean_squared_error')
            cv_metrics = {'cv_rmse': np.mean(np.sqrt(-cv_scores))}
        elif self.task_type == 'binary_classification':
            cv_scores = cross_val_score(self.model, X_prepared, y_prepared, cv=cv, scoring='roc_auc')
            cv_metrics = {'cv_roc_auc': np.mean(cv_scores)}
        elif self.task_type == 'multiclass_classification':
            cv_scores = cross_val_score(self.model, X_prepared, y_prepared, cv=cv, scoring='accuracy')
            cv_metrics = {'cv_accuracy': np.mean(cv_scores)}
        else:
            logger.error(f'Unknown task type {self.task_type}')
            raise Exception('Unknown task type')

        # Сохранение модели + mlflow
        self.status = 'trained' if self.status != 'refitting' else 'retrained'
        self.save()
        
        eval_metric = self.evaluate(X, y)
        cv_metrics.update(eval_metric)

        self.save_model_to_mlflow(cv_metrics)
        self._update_log(cv_metrics)

        logger.info(f'Model {self.model_id} trained with params {self.model_params} and cv_metrics {cv_metrics}')

    def retrain(self, X, y, cv=5, optimize_hyperparameters_flag=False, optimize_hyperparameters_params=None):
        """
        Переобучение модели.
        :param X: Матрица признаков.
        :param y: Вектор целевых значений.
        :param cv: Количество фолдов для кросс-валидации.
        :param optimize_hyperparameters_flag: Флаг для автоматического подбора гиперпараметров.
        :param optimize_hyperparameters_params: Параметры для подбора гиперпараметров под GridSearchCV.
        """
        self.status = 'retrained'
        self.train(X, y, cv, optimize_hyperparameters_flag, optimize_hyperparameters_params)
        logger.info(f'Model {self.model_id} retrained')

    def predict(self, X):
        """
        Предсказание модели.
        :param X: Матрица признаков.
        :return: Вектор предсказаний.
        """
        if self.model is None:
            self.load()

        X_prepared = self.prepare_features(X, None)
        predictions = self.model.predict(X_prepared)
        logger.info(f'Model {self.model_id} made predictions')
        return predictions

    def predict_proba(self, X):
        """
        Предсказание вероятностей модели.
        :param X: Матрица признаков.
        :return: Матрица вероятностей предсказаний.
        """
        if self.model is None:
            self.load()

        if not hasattr(self.model, 'predict_proba'):
            logger.error(f'Model {self.model_id} does not have predict_proba method')
            raise Exception('Model does not have predict_proba method')

        X_prepared = self.prepare_features(X, None)
        predictions_proba = self.model.predict_proba(X_prepared)
        logger.info(f'Model {self.model_id} made probability predictions')
        return predictions_proba

    def save(self):
        """
        Сохранение модели.
        """
        if self.model is None:
            logger.error(f'Model {self.model_id} is not trained')
            raise Exception('Model is not trained')

        joblib.dump(self.model, self.model_path)
        logger.info(f'Model {self.model_id} saved to {self.model_path}')

    def load(self):
        """
        Загрузка модели.
        """
        if self.model is not None:
            logger.info(f'Model {self.model_id} is already loaded')
            return

        if not os.path.exists(self.model_path):
            logger.error(f'Model {self.model_id} not found at {self.model_path}')
            raise Exception('Model not found')

        self.model = joblib.load(self.model_path)
        logger.info(f'Model {self.model_id} loaded from {self.model_path}')

    def delete(self):
        """
        Удаление модели.
        """
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
            logger.info(f'Model {self.model_id} deleted from {self.model_path}')

        if os.path.exists(self.le_path):
            os.remove(self.le_path)
            logger.info(f'LabelEncoder {self.model_id} deleted from {self.le_path}')

        self.status = 'deleted'
        self._update_log()
        self.model = None

    def evaluate(self, X, y, **kwargs):
        """
        Оценка модели.
        :param X: Матрица признаков.
        :param y: Вектор целевых значений.
        :param kwargs: Дополнительные параметры для оценки.
        :return: Словарь с метриками.
        """
        if self.model is None:
            self.load()

        X_prepared = self.prepare_features(X, y)
        y_prepared = self.prepare_taget(y)

        if self.task_type == 'regression':
            y_pred = self.model.predict(X_prepared)
            metrics = reg_model_metric_performance(y_prepared, y_pred, **kwargs)
        elif self.task_type == 'binary_classification':
            y_pred_proba = self.model.predict_proba(X_prepared)
            y_pred = y_pred_proba[:, 1]  # Предсказания для положительного класса
            metrics = binary_model_metric_performance(y_prepared, y_pred, **kwargs)
        elif self.task_type == 'multiclass_classification':
            y_pred_proba = self.model.predict_proba(X_prepared)
            metrics = multiclass_model_metric_performance(y_prepared, y_pred_proba, **kwargs)
        else:
            logger.error(f'Unknown task type {self.task_type}')
            raise Exception('Unknown task type')
            
        logger.info(f'Model {self.model_id} evaluated with metrics {metrics}')
        return metrics

    def start_mlflow_experiment(self):
        """
        Инициализация эксперимента в MLflow.
        """
        try:
            mlflow.set_experiment(self.mlflow_experiment_name)
            logger.info(f'MLflow experiment started: {self.mlflow_experiment_name}')
        except Exception as e:
            logger.error(f'Error starting MLflow experiment: {e}')
            raise

    def log_mlflow_parameters(self):
        """
        Логирование параметров модели в MLflow.
        """
        try:
            mlflow.log_params(self.model_params)
            logger.info(f'MLflow parameters logged: {self.model_params}')
        except Exception as e:
            logger.error(f'Error logging MLflow parameters: {e}')
            raise

    def log_mlflow_metrics(self, metrics):
        """
        Логирование метрик модели в MLflow.
        :param metrics: Словарь с метриками.
        """
        try:
            mlflow.log_metrics(metrics)
            logger.info(f'MLflow metrics logged: {metrics}')
        except Exception as e:
            logger.error(f'Error logging MLflow metrics: {e}')
            raise

    def save_model_to_mlflow(self, metrics=None):
        """
        Сохранение модели в MLflow.
        :param metrics: Метрики модели
        """
        try:
            with mlflow.start_run(run_name=self.model_id):
                mlflow.log_params(self.model_params)
                mlflow.sklearn.log_model(self.model, "model")
                if isinstance(metrics, dict):
                    mlflow.log_metrics(metrics)
                logger.info(f'Model saved to MLflow')
        except Exception as e:
            logger.error(f'Error saving model to MLflow: {e}')
            raise

    def load_model_from_mlflow(self, run_id):
        """
        Загрузка модели из MLflow.
        :param run_id: ID эксперимента в MLflow.
        """
        try:
            self.model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
            logger.info(f'Model loaded from MLflow: run_id={run_id}')
        except Exception as e:
            logger.error(f'Error loading model from MLflow: {e}')
            raise

    def download_dataset_from_minio(self, minio_client, bucket_name, object_name, file_path):
        """
        Скачивание датасета из Minio.
        :param minio_client: Клиент Minio.
        :param bucket_name: Имя бакета.
        :param object_name: Имя объекта в бакете.
        :param file_path: Путь к файлу для сохранения.
        """
        try:
            minio_client.fget_object(bucket_name, object_name, file_path)
            logger.info(f'Dataset downloaded from Minio: {object_name} to {file_path}')
        except Exception as e:
            logger.error(f'Error downloading dataset from Minio: {e}')
            raise