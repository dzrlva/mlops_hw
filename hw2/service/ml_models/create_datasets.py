import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

def create_dataset(task_type, n_numeric, n_categorical, n_samples=100, df=None, n_classes=2):
    """
    Создает датасет на основе sklearn.datasets с заданным количеством числовых и категориальных признаков.
    
    Parameters:
    - task_type: str, тип задачи ('classification' или 'regression')
    - n_numeric: int, количество числовых признаков
    - n_categorical: int, количество категориальных признаков
    - n_samples: int, количество строк в датасете
    - df: pd.DataFrame, DataFrame для хранения результата (по умолчанию создается новый DataFrame)
    
    Returns:
    - pd.DataFrame с сгенерированными данными
    """
    if task_type not in ['binary_classification', 'regression', 'multiclass_classification', 'classification']:
        raise ValueError("task_type должен быть 'classification' или 'regression'")
    
    if df is None:
        df = pd.DataFrame()
    
    # Генерация числовых признаков
    if 'classification' in task_type:
        X_numeric, y = make_classification(n_samples=n_samples, 
            n_features=n_numeric, 
            n_redundant=0,
            n_repeated=0,
            n_informative=n_numeric,
            n_clusters_per_class=1,
            n_classes=n_classes)
    elif task_type == 'regression':
        X_numeric, y = make_regression(n_samples=n_samples, 
            n_features=n_numeric, 
            n_informative=n_numeric, 
            noise=0.1)
    
    # Добавление числовых признаков в DataFrame
    for i in range(n_numeric):
        df[f'feature_{i}'] = X_numeric[:, i]
    
    # Генерация категориальных признаков
    if n_categorical > 0:

        # Добавление категориальных признаков в DataFrame
        for j in range(n_categorical):
            n_categories = np.random.randint(2,10)
            p = np.random.rand(n_categories)
            X_categorical = np.random.choice(np.arange(n_categories), p=p / p.sum(), size=n_samples)
            df[f'category_{j}'] = X_categorical.astype(str)
    
    # Добавление целевой переменной в DataFrame
    df['target'] = y
    
    return df