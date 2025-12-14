# syntax=docker/dockerfile:1
FROM python:3.9-slim-buster

WORKDIR /app

# Copy common requirements and install them
COPY services/common/requirements.txt ./requirements.txt
RUN pip install --default-timeout=600 --no-cache-dir -r requirements.txt

# Ensure data directory exists for SQLite persistence
RUN mkdir -p /app/data

# Copy the service-specific code
COPY services/api_gateway/ ./services/api_gateway/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
