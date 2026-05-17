FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.4.1 \
    PORT=8050

WORKDIR /app

RUN python -m pip install --upgrade pip \
    && python -m pip install "poetry==${POETRY_VERSION}" \
    && poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-interaction

COPY . .

EXPOSE 8050

CMD ["gunicorn", "app:server", "--config", "gunicorn.conf.py"]
