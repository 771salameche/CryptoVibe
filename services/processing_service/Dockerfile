# syntax=docker/dockerfile:1
FROM python:3.9-slim-buster

WORKDIR /app

# Copy common requirements and install them
COPY services/common/requirements.txt ./requirements.txt
RUN pip install --default-timeout=600 --no-cache-dir -r requirements.txt

# Copy the service-specific code
COPY services/processing_service/ ./services/processing_service/

# (Optional: Add an entrypoint or command if the service is ready to run)
CMD ["python", "main.py"]
