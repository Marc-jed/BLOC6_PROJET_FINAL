# Utiliser une image Python
FROM python:3.9

# Définir le dossier de travail dans le conteneur
WORKDIR /app

# Copier requirements.txt en premier (pour tirer parti du cache)
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers du projet dans le conteneur
COPY . .

# Exposer le port par défaut de Streamlit
EXPOSE 8501

# Lancer l'application Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]