# app/utils.py
import os
import importlib.util
from pathlib import Path
from .ml_models.base_model import BaseModel

def find_model_files(models_dir):
    """
    Поиск файлов моделей в указанной директории.
    :param models_dir: Путь к директории с моделями.
    :return: Список названий файлов моделей.
    """
    model_files = []
    for file in os.listdir(models_dir):
        if file.endswith('_model.py') and file != 'base_model.py':
            model_files.append(file)
    return model_files

def import_module_from_file(file_path):
    """
    Динамический импорт модуля из файла.
    :param file_path: Путь к файлу.
    :return: Импортированный модуль.
    """
    module_name = Path(file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_model_classes(module):
    """
    Получение классов моделей из модуля.
    :param module: Импортированный модуль.
    :return: Список классов моделей.
    """
    model_classes = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, BaseModel) and attr is not BaseModel:
            model_classes.append(attr)
    return model_classes