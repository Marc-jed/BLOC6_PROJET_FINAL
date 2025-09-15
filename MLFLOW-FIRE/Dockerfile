FROM python:3.9-slim

WORKDIR /home/app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Exposition du port MLflow
EXPOSE 5000

# Définir les variables d'environnement pour AWS et MLflow
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV BACKEND_STORE_URI=$BACKEND_STORE_URI
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=$MLFLOW_DEFAULT_ARTIFACT_ROOT


# Commande pour démarrer le serveur MLflow
CMD mlflow server -p $PORT \
    --host 0.0.0.0 \
    --backend-store-uri $BACKEND_STORE_URI \
    --default-artifact-root $MLFLOW_DEFAULT_ARTIFACT_ROOT   