FROM python:3.11-slim

WORKDIR /app

COPY inference/requirements.txt ./inference/requirements.txt
RUN pip install --no-cache-dir -r ./inference/requirements.txt

COPY inference ./inference
COPY project/metadata ./project/metadata
COPY project/data/processed/bitcoin_inference.csv ./project/data/processed/bitcoin_inference.csv

ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

EXPOSE 8000

CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
