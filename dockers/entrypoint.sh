#!/bin/bash

set -e

env

uv pip list

python src/manage.py create_vector_extension

echo "Running database migrations..."
python src/manage.py makemigrations
python src/manage.py migrate

# Create a custom management command to create the vector extension.
# I will provide instructions on how to do this.
python src/manage.py create_vector_extension

echo "Starting Gunicorn with Uvicorn workers..."
# Replace 'your_project_name' with your Django project's name.
gunicorn --chdir /app/src/ config.asgi:application \
  --bind 0.0.0.0:${APP_PORT:-8000} --workers ${APP_WORKERS:-1} \
  --worker-class uvicorn.workers.UvicornWorker