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
from catboost import CatBoostRegressor, CatBoostClassifier, Pool

from base_model import BaseModel

# Настройка логгирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Функции для оценки моделей уже определены в вашем коде

class CatBoostModel(BaseModel):
    """
    Класс для моделей CatBoost.
    """
    def __init__(self, model_id=None, model_params=None, task_type='regression'):
        super().__init__(model_id, model_params, task_type)

    def prepare_features(self, X):
        """
        Подготовка признаков.
        :param X: Матрица признаков.
        :return: Подготовленные признаки.
        """
        return X

    def create_pool(self, X, y):
        """
        Создание Pool для CatBoost.
        :param X: Матрица признаков.
        :param y: Вектор целевых значений.
        :return: Pool.
        """
        # Определение типов признаков
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Создание Pool
        pool = Pool(X, y, cat_features=categorical_features)
        return pool

    def fit(self, X, y):
        """
        Обучение модели.
        :param X: Матрица признаков.
        :param y: Вектор целевых значений.
        """
        pool = self.create_pool(X, y)

        if self.task_type == 'regression':
            self.model = CatBoostRegressor(**self.model_params)
        elif 'classification' in self.task_type:
            self.model = CatBoostClassifier(**self.model_params)
        else:
            logging.error(f'Unknown task type {self.task_type}')
            raise Exception('Unknown task type')

        self.model.fit(pool)
        logging.info(f'Model {self.model_id} fitted')

    def optimize_hyperparameters(self, X, y, optimize_hyperparameters_params):
        """
        Подбор гиперпараметров модели.
        :param X: Матрица признаков.
        :param y: Вектор целевых значений.
        :param optimize_hyperparameters_params: Параметры для подбора гиперпараметров под GridSearchCV.
        :return: Оптимальные гиперпараметры.
        """
        pool = self.create_pool(X, y)

        if self.task_type == 'regression':
            model = CatBoostRegressor(**self.model_params)
        elif 'classification' in self.task_type:
            model = CatBoostClassifier(self.model_params)
        else:
            logging.error(f'Unknown task type {self.task_type}')
            raise Exception('Unknown task type')

        grid_search = model.grid_search(optimize_hyperparameters_params, pool)
        best_params = model.get_all_params()
        logging.info(f'Model {self.model_id} optimized with params {best_params}')
        return best_params