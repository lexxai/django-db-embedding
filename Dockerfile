ARG PYTHON_VER=3.14

FROM python:${PYTHON_VER} AS build

# Install uv
# RUN pip install uv
# Use an official image to get the uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set work directory
WORKDIR /app

# Install dependencies
COPY "pyproject.toml" "uv.lock" .
RUN uv sync --locked

FROM python:${PYTHON_VER}-slim

# Set work directory
WORKDIR /app
COPY --from=build /app/.venv /app/.venv

# Copy project
COPY --chmod=+x ./dockers/entrypoint.sh .
COPY ./src/db_embedding/ .

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH=/app/.venv/bin/:$PATH

ENTRYPOINT ["/app/entrypoint.sh"]

