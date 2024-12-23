#!/bin/bash

pip install dvc[s3]
git config --global user.name service_dvc
git config --global user.email alex.kormishenkov@gmail.com
git init
dvc remote modify --local myremote access_key_id $MINIO_ACCESS_KEY
dvc remote modify --local myremote secret_access_key $MINIO_SECRET_KEY
dvc remote modify myremote endpointurl http://$MINIO_SERVER:9000

dvc pull 
echo "DVC all set!"