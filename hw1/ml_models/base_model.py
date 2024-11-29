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

from task_metrics import reg_model_metric_performance, binary_model_metric_performance, multiclass_model_metric_performance

# Настройка логгирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseModel(ABC):
    """
    Базовый класс для модели машинного обучения.
    """
    def __init__(self, model_id=None, model_params=None, task_type='regression'):
        """
        Инициализация модели.
        :param model_id: Уникальный идентификатор модели.
        :param model_params: Параметры модели.
        :param task_type: Тип задачи (regression, binary_classification, multiclass_classification).
        """
        self.model_id = model_id if model_id else str(uuid4())
        self.model_params = model_params if model_params else {}
        self.model = None
        self.status = 'initialized'
        self.model_path = os.path.join('../models', f'{self.model_id}.joblib')
        self.log_path = 'models_log.csv'
        self.task_type = task_type
        # Создаем директорию для моделей, если она не существует
        if not os.path.exists('../models'):
            os.makedirs('../models')
        # Создаем журнал моделей, если он не существует
        if not os.path.exists(self.log_path):
            log_df = pd.DataFrame(columns=['model_id', 'model_params', 'status', 'status_time', 'model_path', 'task_type', 'cv_metrics'])
            log_df.to_csv(self.log_path, index=False)
        # Записываем информацию о модели в журнал
        self._update_log()
        logging.info(f'Model {self.model_id} initialized with params {self.model_params} for task {self.task_type}')

    def _update_log(self, cv_metrics=None):
        """
        Обновление журнала моделей.
        :param cv_metrics: Метрики кросс-валидации.
        """
        new_log = pd.DataFrame({
            'model_id': [self.model_id],
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
    def prepare_features(self, X):
        """
        Подготовка признаков.
        :param X: Матрица признаков.
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

    def train(self, X, y, cv=5, optimize_hyperparameters_flag=False, optimize_hyperparameters_params=None):
        """
        Обучение модели с оценкой метрик на кросс-валидации и подбором гиперпараметров.
        :param X: Матрица признаков.
        :param y: Вектор целевых значений.
        :param cv: Количество фолдов для кросс-валидации.
        :param optimize_hyperparameters_flag: Флаг для автоматического подбора гиперпараметров.
        :param optimize_hyperparameters_params: Параметры для подбора гиперпараметров под GridSearchCV.
        """
        X_prepared = self.prepare_features(X)
        # Подбор гиперпараметров, если флаг установлен
        if optimize_hyperparameters_flag:
            self.model_params = self.optimize_hyperparameters(X_prepared, y, optimize_hyperparameters_params)
            logging.info(f'Model {self.model_id} optimized with params {self.model_params}')
        self.fit(X_prepared, y)
        # Оценка метрик на кросс-валидации
        if self.task_type == 'regression':
            cv_scores = cross_val_score(self.model, X_prepared, y, cv=cv, scoring='neg_mean_squared_error')
            cv_metrics = {'cv_rmse': np.mean(np.sqrt(-cv_scores))}
        elif self.task_type == 'binary_classification':
            cv_scores = cross_val_score(self.model, X_prepared, y, cv=cv, scoring='roc_auc')
            cv_metrics = {'cv_roc_auc': np.mean(cv_scores)}
        elif self.task_type == 'multiclass_classification':
            cv_scores = cross_val_score(self.model, X_prepared, y, cv=cv, scoring='accuracy')
            cv_metrics = {'cv_accuracy': np.mean(cv_scores)}
        else:
            logging.error(f'Unknown task type {self.task_type}')
            raise Exception('Unknown task type')
        self.status = 'trained'
        self.save()
        self._update_log(cv_metrics)
        logging.info(f'Model {self.model_id} trained with params {self.model_params} and cv_metrics {cv_metrics}')

    def retrain(self, X, y, optimize_hyperparameters_flag=False, optimize_hyperparameters_params=None):
        """
        Переобучение модели.
        :param X: Матрица признаков.
        :param y: Вектор целевых значений.
        :param optimize_hyperparameters_flag: Флаг для автоматического подбора гиперпараметров.
        :param optimize_hyperparameters_params: Параметры для подбора гиперпараметров под GridSearchCV.
        """
        self.train(X, y, optimize_hyperparameters_flag, optimize_hyperparameters_params)
        self.status = 'retrained'
        self.save()
        logging.info(f'Model {self.model_id} retrained')

    def predict(self, X):
        """
        Предсказание модели.
        :param X: Матрица признаков.
        :return: Вектор предсказаний.
        """
        if self.model is None:
            self.load()
        X_prepared = self.prepare_features(X)
        predictions = self.model.predict(X_prepared)
        logging.info(f'Model {self.model_id} made predictions')
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
            logging.error(f'Model {self.model_id} does not have predict_proba method')
            raise Exception('Model does not have predict_proba method')
        X_prepared = self.prepare_features(X)
        predictions_proba = self.model.predict_proba(X_prepared)
        logging.info(f'Model {self.model_id} made probability predictions')
        return predictions_proba

    def save(self):
        """
        Сохранение модели.
        """
        if self.model is None:
            logging.error(f'Model {self.model_id} is not trained')
            raise Exception('Model is not trained')
        joblib.dump(self.model, self.model_path)
        self._update_log()
        logging.info(f'Model {self.model_id} saved to {self.model_path}')

    def load(self):
        """
        Загрузка модели.
        """
        if self.model is not None:
            logging.info(f'Model {self.model_id} is already loaded')
            return
        if not os.path.exists(self.model_path):
            logging.error(f'Model {self.model_id} not found at {self.model_path}')
            raise Exception('Model not found')
        self.model = joblib.load(self.model_path)
        self.status = 'loaded'
        self._update_log()
        logging.info(f'Model {self.model_id} loaded from {self.model_path}')

    def delete(self):
        """
        Удаление модели.
        """
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
            logging.info(f'Model {self.model_id} deleted from {self.model_path}')
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
        X_prepared = self.prepare_features(X)
        if self.task_type == 'regression':
            y_pred = self.model.predict(X_prepared)
            metrics = reg_model_metric_performance(y, y_pred, **kwargs)
        elif self.task_type == 'binary_classification':
            y_pred_proba = self.model.predict_proba(X_prepared)
            y_pred = y_pred_proba[:, 1]  # Предсказания для положительного класса
            metrics = binary_model_metric_performance(y, y_pred, **kwargs)
        elif self.task_type == 'multiclass_classification':
            y_pred_proba = self.model.predict_proba(X_prepared)
            metrics = multiclass_model_metric_performance(y, y_pred_proba, **kwargs)
        else:
            logging.error(f'Unknown task type {self.task_type}')
            raise Exception('Unknown task type')
        logging.info(f'Model {self.model_id} evaluated with metrics {metrics}')
        return metrics