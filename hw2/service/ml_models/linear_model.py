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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from ml_models.base_model import BaseModel

# Настройка логгирования
logging.basicConfig(filename='./ml_service_logs', filemode='a', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('./ml_service_logs')

class LinearModel(BaseModel):
    """
    Класс для линейных моделей машинного обучения.
    """
    def __init__(self, model_id=None, model_description=None, model_params=None, task_type='regression', mlflow_experiment_name=None):
        super().__init__(model_id, model_description, model_params, task_type, mlflow_experiment_name)
        self.pipeline_path = os.path.join('pipelines', f'{self.model_id}_pipeline.joblib')
        # Создаем директорию для пайплайнов, если она не существует
        if not os.path.exists('pipelines'):
            os.makedirs('pipelines')

    def features_pipeline(self, X, y):
        """
        Создание, обучение и сохранение пайплайна для обработки данных.
        :param X: Матрица признаков.
        :param y: Вектор целевых значений.
        :return: Пайплайн.
        """
        # Определение типов признаков
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        # Создание трансформеров
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore',drop='first'))
        ])

        # Создание пайплайна
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

        # Обучение пайплайна
        pipeline.fit(X, y)

        # Сохранение пайплайна
        joblib.dump(pipeline, self.pipeline_path)
        logger.info(f'Pipeline {self.model_id} saved to {self.pipeline_path}')
        return pipeline

    def prepare_features(self, X, y):
        """
        Подготовка признаков с использованием пайплайна.
        :param X: Матрица признаков.
        :param y: Вектор целевых значений.
        :return: Подготовленные признаки.
        """
        if not os.path.exists(self.pipeline_path):
            logger.warning(f'Pipeline {self.model_id} not found at {self.pipeline_path}')
            logger.info(f'Pipeline {self.model_id} start fitting')
            self.features_pipeline(X, y)

        pipeline = joblib.load(self.pipeline_path)
        logger.info(f'Pipeline {self.model_id} loaded from {self.pipeline_path}')

        X_prepared = pipeline.transform(X)
        return X_prepared

    def fit(self, X, y):
        """
        Обучение модели.
        :param X: Матрица признаков.
        :param y: Вектор целевых значений.
        """
        if self.task_type == 'regression':
            self.model = LinearRegression(**self.model_params)
        elif 'classification' in self.task_type:
            self.model = LogisticRegression(**self.model_params)
        else:
            logger.error(f'Unknown task type {self.task_type}')
            raise Exception('Unknown task type')

        self.model.fit(X, y)
        logger.info(f'Model {self.model_id} fitted')

    def optimize_hyperparameters(self, X, y, optimize_hyperparameters_params):
        """
        Подбор гиперпараметров модели.
        :param X: Матрица признаков.
        :param y: Вектор целевых значений.
        :return: Оптимальные гиперпараметры.
        :param optimize_hyperparameters_params: Параметры для подбора гиперпараметров под GridSearchCV.
        """
        if self.task_type == 'regression':
            model = LinearRegression(**self.model_params)
        elif 'classification' in self.task_type:
            model = LogisticRegression(**self.model_params)
        else:
            logger.error(f'Unknown task type {self.task_type}')
            raise Exception('Unknown task type')

        if self.task_type == 'regression':
            scoring_metric = 'neg_mean_squared_error'
        elif self.task_type == 'binary_classification':
            scoring_metric = 'roc_auc'
        elif self.task_type == 'multiclass_classification':
            scoring_metric = 'accuracy'
        else:
            logger.error(f'Unknown task type {self.task_type}')
            raise Exception('Unknown task type')

        grid_search = GridSearchCV(model, 
            optimize_hyperparameters_params, 
            cv=5, 
            scoring=scoring_metric)
        grid_search.fit(X, y)
        best_params = grid_search.best_estimator_.get_params()
        logger.info(f'Model {self.model_id} optimized with params {best_params}')
        return best_params

    def delete(self):
        """
        Удаление модели и пайплайна.
        """
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
            logger.info(f'Model {self.model_id} deleted from {self.model_path}')
        self.status = 'deleted'
        self._update_log()
        self.model = None

        if os.path.exists(self.pipeline_path):
            os.remove(self.pipeline_path)
            logger.info(f'Pipeline {self.model_id} deleted from {self.pipeline_path}')

        if os.path.exists(self.le_path):
            os.remove(self.le_path)
            logger.info(f'LabelEncoder {self.model_id} deleted from {self.le_path}')