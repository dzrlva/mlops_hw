import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    median_absolute_error, max_error, mean_absolute_percentage_error, mean_absolute_error,
    mean_squared_error, mean_squared_log_error, r2_score, explained_variance_score,
    mean_gamma_deviance, mean_poisson_deviance, accuracy_score, precision_score,
    recall_score, fbeta_score, roc_auc_score, average_precision_score
)

def reg_model_metric_performance(y_true, y_pred, **kwargs):
    '''
    Функция, которая оценивает регрессионную модель по основным метрикам
    ----------------------------------------------------------------------
    Parametrs:
    y_true: np.array/pd.Series
        вектор целевой переменной
    y_pred: np.array/pd.Series
        вектор предсказаний
    round_n: int
        до какого порядка округляем
    -------------------------------------------------------------------------
    Return:
        dict with metrics
    '''
    round_n = kwargs.get('round_n', None)
    # округлим до n порядка
    if isinstance(round_n, int):
        y_pred = round_n * np.round(y_pred / round_n)
    # расчет основных метрик
    reg_metrics = {
        'N': len(y_pred),
        'std_target': y_true.std(), 'mean_target': y_true.mean(), 'median_target': np.median(y_true),
        'std_pred': y_pred.std(), 'mean_pred': y_pred.mean(), 'median_pred': np.median(y_pred),
        'medianae': median_absolute_error(y_true, y_pred), 'maxae': max_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred), 'mae': mean_absolute_error(y_true, y_pred),
        'rmse': mean_squared_error(y_true, y_pred, squared=False), 'rmsle': mean_squared_log_error(y_true, y_pred, squared=False),
        'R2': r2_score(y_true, y_pred), 'explained_variance': explained_variance_score(y_true, y_pred),
        'gamma': mean_gamma_deviance(y_true, y_pred), 'poisson': mean_poisson_deviance(y_true, y_pred),
        'medianae / median': median_absolute_error(y_true, y_pred) / np.median(y_true), 
        'mae / mean': mean_absolute_error(y_true, y_pred) / np.mean(y_true)
    }
    return reg_metrics

def binary_model_metric_performance(y_true, y_pred, **kwargs):
    '''
    Функция, которая оценивает модель бинарной классификации по основным метрикам
    ----------------------------------------------------------------------
    Parametrs:
    y_true: np.array/pd.Series
        вектор целевой переменной
    y_pred: np.array/pd.Series
        вектор предсказаний
    threshold: float
        порог для бинаризации предсказаний модели
    -------------------------------------------------------------------------
    Return:
        dict with metrics
    '''
    threshold = kwargs.get('threshold', 0.5)
    pred = (y_pred >= threshold).astype(int)
    target_distrib = np.unique(y_true, return_counts=True)
    pred_distrib = np.unique(pred, return_counts=True)
    target_classes = {f"target_label_count_{target_distrib[0][i]}": target_distrib[1][i] for i in range(len(target_distrib[0]))}
    pred_classes = {f"pred_label_count_{pred_distrib[0][i]}": pred_distrib[1][i] for i in range(len(pred_distrib[0]))}
    binary_metrics = {
        'N': len(pred),
        'accuracy': accuracy_score(y_true, pred),
        'precision': precision_score(y_true, pred),
        'recall': recall_score(y_true, pred),
        'f1': fbeta_score(y_true, pred, beta=1),
        'roc-auc': roc_auc_score(y_true, pred),
        'pr-auc': average_precision_score(y_true, pred),
    }
    binary_metrics.update(target_classes)
    binary_metrics.update(pred_classes)
    return binary_metrics

def multiclass_model_metric_performance(y_true, y_pred_proba, **kwargs):
    '''
    Функция, которая оценивает модель мультиклассовой классификации по основным метрикам
    ----------------------------------------------------------------------
    Parametrs:
    y_true: np.array/pd.Series
        вектор целевой переменной
    y_pred_proba: np.array/pd.DataFrame
        матрица вероятностей предсказаний модели
    -------------------------------------------------------------------------
    Return:
        dict with metrics
    '''
    pred = np.argmax(y_pred_proba, axis=1)
    target_distrib = np.unique(y_true, return_counts=True)
    pred_distrib = np.unique(pred, return_counts=True)
    target_classes = {f"target_label_count_{target_distrib[0][i]}": target_distrib[1][i] for i in range(len(target_distrib[0]))}
    pred_classes = {f"pred_label_count_{pred_distrib[0][i]}": pred_distrib[1][i] for i in range(len(pred_distrib[0]))}
    multiclass_metrics = {
        'N': len(pred),
        'accuracy': accuracy_score(y_true, pred),
        'precision_micro': precision_score(y_true, pred, average='micro'),
        'precision_macro': precision_score(y_true, pred, average='macro'),
        'recall_micro': recall_score(y_true, pred, average='micro'),
        'recall_macro': recall_score(y_true, pred, average='macro'),
        'f1_micro': fbeta_score(y_true, pred, average='micro', beta=1),
        'f1_macro': fbeta_score(y_true, pred, average='macro', beta=1)
    }
    multiclass_metrics.update(target_classes)
    multiclass_metrics.update(pred_classes)
    return multiclass_metrics