Структура проекта:
ml_service/
├── ml_models/
│   ├── __init__.py
│   ├── base_model.py
│   ├── catboost_model.py
│   ├── linear_model.py
│   ├── tree_model.py
│   └── task_metrics.py
├── app/
│   ├── __init__.py
│   ├── main.py - основной функционал rest service на fastapi 
│   ├── utils.py - утилитные функции для поиска доступных классов моделей
│   └── pydantic_models.py - Структура запросов к сервису
├── pyproject.toml
└── README.md
