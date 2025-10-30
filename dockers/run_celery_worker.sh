#!/bin/bash

set -e
echo "Running celery worker..."
cd src/
celery -A config worker -l info --pool=solo --concurrency=1