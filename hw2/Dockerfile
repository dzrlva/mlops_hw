# Dockerfile для FastAPI приложения

# Используем официальный образ Python
FROM python:3.12-slim

# Устанавливаем рабочую директорию
WORKDIR /service

# Копируем файлы зависимостей
COPY service/pyproject.toml ./

# Устанавливаем Poetry
RUN pip install poetry

# Установка Git
RUN apt-get update && apt-get install -y git

# Устанавливаем зависимости
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# Копируем остальные файлы приложения
COPY service/ .

# Устанавливаем переменные окружения
ENV PYTHONPATH=/service

# Экспонируем порт
EXPOSE 8000

# Запуск приложения
CMD ["/bin/sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000"]