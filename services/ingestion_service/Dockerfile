# syntax=docker/dockerfile:1
FROM python:3.9-slim-buster

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=1200 \
    PIP_RETRIES=5

# Copy common requirements and install them
COPY services/common/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install --default-timeout=1200 --no-cache-dir -r requirements.txt

# Copy the service-specific code
COPY services/ingestion_service/ ./services/ingestion_service/

CMD ["python", "main.py"]
