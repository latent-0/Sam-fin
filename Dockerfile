FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uvicorn
COPY fa ./fa
COPY services ./services
COPY frontend ./frontend
COPY artifacts ./artifacts
COPY paper/results ./paper/results
ENV PYTHONUNBUFFERED=1
# SERVICE is the module name under services/, e.g. gateway or income_svc
CMD ["sh", "-c", "uvicorn services.${SERVICE}:app --host 0.0.0.0 --port ${PORT}"]
