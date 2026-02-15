FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs results

ENV PYTHONUNBUFFERED=1
ENV RAY_DEDUP_LOGS=0
ENV PYTHONWARNINGS=ignore

CMD ["flwr", "run", "."]