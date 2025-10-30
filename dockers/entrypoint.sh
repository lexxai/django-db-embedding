#!/bin/bash

set -e

env

uv pip list

python src/manage.py create_vector_extension

echo "Running database migrations..."
python src/manage.py makemigrations
python src/manage.py migrate

python src/manage.py createsuperuser --username admin  --noinput || true


echo "Starting Gunicorn with Uvicorn workers..."
gunicorn --chdir /app/src/ config.asgi:application \
  --bind 0.0.0.0:${APP_PORT:-8000} --workers ${APP_WORKERS:-1} \
  --worker-class uvicorn.workers.UvicornWorker