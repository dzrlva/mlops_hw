#!/bin/bash

# Установка DVC с поддержкой S3
pip install dvc[s3]

# Настройка глобальных настроек для DVC
git config --global user.name "service_dvc"
git config --global user.email "alex.kormishenkov@gmail.com"

# Создание директории data, если она не существует
mkdir -p ./upload_data

# Инициализация DVC в директории data
cd ./upload_data
git init
dvc init
echo "DVC initialized in ./upload_data"

# Настройка удаленного хранилища DVC для MinIO
dvc remote add -d ./upload_data s3://$MINIO_BUCKET_NAME
dvc remote modify ./upload_data access_key_id $MINIO_ACCESS_KEY
dvc remote modify ./upload_data secret_access_key $MINIO_SECRET_KEY
dvc remote modify ./upload_data endpointurl http://$MINIO_ENDPOINT

# Проверка инициализации
dvc remote list
echo "DVC remote configured for minio"

# Возврат в корневую директорию
cd ..