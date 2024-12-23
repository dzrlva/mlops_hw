# REST Сервис для Обучения Машинного Обучения

## Описание

Этот REST сервис позволяет загружать, обучать, оценивать, предсказывать и управлять различными моделями машинного обучения. Сервис реализован на Python с использованием фреймворка FastAPI и включает в себя несколько типов моделей, таких как линейные модели, модели на основе CatBoost и деревья решений.

## Структура Проекта
mlops_hw/
├── app/
│   ├── main.py                # Основной файл сервиса FastAPI
│   ├── pydantic_models.py     # Определения моделей запросов и ответов
│   ├── streamlit.py           # Интерфейс для взаимодействия с сервисом через Streamlit
│   ├── utils.py               # Вспомогательные функции для работы с моделями
│   └── __init__.py            # Инициализация пакета app
├── ml_models/
│   ├── base_model.py          # Базовый класс для моделей машинного обучения
│   ├── catboost_model.py      # Реализация модели на основе CatBoost
│   ├── linear_model.py        # Реализация линейной модели
│   ├── tree_model.py          # Реализация модели на основе деревьев решений
│   ├── task_metrics.py        # Функции для оценки моделей
│   ├── create_datasets.py     # Функции для создания синтетических данных
│   └── __init__.py            # Инициализация пакета ml_models
├── tests/
│   └── test_service.py        # Тесты для сервиса
├── pyproject.toml             # Файл конфигурации poetry
└── README.md                  # Документация

## Запуск
0. Поставить poetry
1. Клонирование репозитория
2. Перейти в папке репозитория и выполнить команды: 
 ```bash
    poetry install
```
3. Активировать сервис и запустить интерфейс Streamlit (опционально):
```bash
    poetry run uvicorn app.main:app --reload
```
Сервис будет доступен по адресу `http://127.0.0.1:8000`.
```bash
    poetry run streamlit run app/streamlit.py
```

Интерфейс будет доступен по адресу `http://127.0.0.1:8501`.

## Функциональность

1. **Получение списка моделей:**

    - **URL:** `/models`
    - **Метод:** `GET`
    - **Описание:** Возвращает список доступных моделей.
    - **Пример запроса:**

        ```bash
        curl http://127.0.0.1:8000/models
        ```

2. **Получение статуса сервиса:**

    - **URL:** `/status`
    - **Метод:** `GET`
    - **Описание:** Возвращает статус сервиса.
    - **Пример запроса:**

        ```bash
        curl http://127.0.0.1:8000/status
        ```

3. **Получение информации о моделях:**

    - **URL:** `/models_info`
    - **Метод:** `GET`
    - **Описание:** Возвращает информацию о всех моделях.
    - **Пример запроса:**

        ```bash
        curl http://127.0.0.1:8000/models_info
        ```

4. **Обучение модели:**

    - **URL:** `/train`
    - **Метод:** `POST`
    - **Описание:** Обучает модель на предоставленных данных.
    - **Пример запроса:**

        ```bash
        curl -X POST http://127.0.0.1:8000/train -H "Content-Type: application/json" -d '{
            "model_type": "LinearModel",
            "task_type": "regression",
            "feature_names": ["feature_0", "feature_1"],
            "features": [[1.0, 2.0], [3.0, 4.0]],
            "target": [0.0, 1.0],
            "cv": 5,
            "optimize_hyperparameters_flag": false,
            "optimize_hyperparameters_params": {}
        }'
        ```

5. **Оценка модели:**

    - **URL:** `/evaluate`
    - **Метод:** `POST`
    - **Описание:** Оценивает модель на предоставленных данных.
    - **Пример запроса:**
    - **Model id** конечно надо брать после вызова models_info или сохранить после обучения

        ```bash
        curl -X POST http://127.0.0.1:8000/evaluate -H "Content-Type: application/json" -d '{
            "model_id": "model_id",
            "feature_names": ["feature_0", "feature_1"],
            "features": [[1.0, 2.0], [3.0, 4.0]],
            "target": [0.0, 1.0]
        }'
        ```

6. **Предсказание модели:**

    - **URL:** `/predict`
    - **Метод:** `POST`
    - **Описание:** Делает предсказания с помощью модели.
    - **Пример запроса:**
    - **Model id** конечно надо брать после вызова models_info или сохранить после обучения

        ```bash
        curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
            "model_id": "model_id",
            "prediction_type": "predict",
            "feature_names": ["feature_0", "feature_1"],
            "features": [[1.0, 2.0], [3.0, 4.0]]
        }'
        ```

7. **Переобучение модели:**

    - **URL:** `/retrain`
    - **Метод:** `POST`
    - **Описание:** Переобучает модель на новых данных.
    - **Пример запроса:**
    - **Model id** конечно надо брать после вызова models_info или сохранить после обучения

        ```bash
        curl -X POST http://127.0.0.1:8000/retrain -H "Content-Type: application/json" -d '{
            "model_id": "model_id",
            "task_type": "regression",
            "feature_names": ["feature_0", "feature_1"],
            "features": [[1.0, 2.0], [3.0, 4.0]],
            "target": [0.0, 1.0],
            "cv": 5,
            "optimize_hyperparameters_flag": false,
            "optimize_hyperparameters_params": {}
        }'
        ```

8. **Удаление модели:**

    - **URL:** `/delete/{model_id}`
    - **Метод:** `DELETE`
    - **Описание:** Удаляет модель.
    - **Пример запроса:**
    - **Model id** конечно надо брать после вызова models_info или сохранить после обучения

        ```bash
        curl -X DELETE http://127.0.0.1:8000/delete/model_id
        ```
## Тестирование
Для тестирования сервиса используйте файл test_service.py. Убедитесь, что сервис запущен перед выполнением тестов. Запуск: ```poetry run pytest tests/test_service.py```

## Контакты
Авторы: Озерова Дарья, Кормишенков Александр  
Tg: @dzrlva, @digitkorm
