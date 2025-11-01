#!/bin/bash

set -e
echo "Running create vector extension..."
python src/manage.py create_vector_extension || true
echo "Running collect static..."
python src/manage.py collectstatic --noinput || true
echo "Running database migrations..."
python src/manage.py makemigrations --noinput && python src/manage.py migrate --noinput || true
echo "Running database setup search indexes..."
python src/manage.py setup_search_indexes || true
echo "Running creating superuser..."
python src/manage.py createsuperuser --username admin  --noinput || true


echo "Starting Gunicorn with Uvicorn workers..."
gunicorn --chdir /app/src/ config.asgi:application \
  --bind 0.0.0.0:${APP_PORT:-8000} --workers ${APP_WORKERS:-1} \
  --worker-class uvicorn.workers.UvicornWorker