# syntax=docker/dockerfile:1
FROM python:3.9-slim-buster

WORKDIR /app

# Copy common requirements and install them
COPY services/common/requirements.txt ./requirements.txt
RUN pip install --default-timeout=600 --no-cache-dir -r requirements.txt

# Copy the service-specific code
COPY services/ingestion_service/ ./services/ingestion_service/

CMD ["python", "main.py"]
