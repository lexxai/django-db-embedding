# Use an official Python runtime as a parent image
FROM python:3.14-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install uv
RUN pip install uv

# Set work directory
WORKDIR /app

# Install dependencies
COPY "pyproject.toml" "uv.lock" .
RUN uv sync --locked

# Copy project
COPY ./src ./src

