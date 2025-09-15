🔥 Prédiction du Risque de Feu en Corse à partir de Données Météo
🧠 Objectif

Ce projet vise à prédire le risque d'incendie (feu) dans le temps pour chaque zone géographique de la Corse, en s’appuyant sur un modèle de survie basé sur des données météorologiques et des données d’historique d’incendies.

🗂️ Données utilisées
🔸 Données d’incendies (BDIFF)

Source : BDIFF - Base de Données des Incendies de Forêt en France

Période : 2006 à 2024

Variables :

Date et lieu du feu

Localisation (commune, latitude, longitude)

🔸 Données météorologiques

Source : Météo-France

Données quotidiennes par station météo en Corse

Variables :

Température, humidité, vent, précipitations, etc.

Données synchronisées avec les dates et localisations des feux

⚙️ Modélisation
📌 Problématique

Estimer la probabilité qu’un feu se déclenche dans une zone donnée à un horizon t (7j, 30j, 60j, 90j, 180j), en fonction des conditions météo récentes.

🔍 Modèle principal

XGBoost Regressor avec l’objectif survival:cox (modèle de survie)

Pipeline de traitement avec :

Imputation des données manquantes (SimpleImputer)

Standardisation (StandardScaler)

Apprentissage supervisé à partir de :

event = feu ou non

duration = temps d’attente jusqu’au feu

🔬 Modèle de base de risque

Dans les modèles de survie, on a besoin de savoir quelle est la probabilité qu’un événement se produise au fil du temps, même sans information particulière sur une zone.

C’est ce qu’on appelle le risque de base :

Il décrit l’évolution du risque moyen au fil du temps (par exemple : quelle est la tendance générale d’apparition des feux en Corse, toutes zones confondues ?).

Ce risque de base est ensuite ajusté avec les conditions locales (température, pluie, vent, etc.) prédites par notre modèle XGBoost.

Concrètement :

Le risque de base agit comme une courbe de référence.

Chaque zone voit sa probabilité de feu augmenter ou diminuer en fonction de ses propres caractéristiques météo.

Cela permet d’obtenir des probabilités de feu cohérentes dans le temps, plutôt qu’une simple prédiction ponctuelle.

🗺️ Visualisation
📍 Carte interactive

Affichage du risque de feu par zone sur une carte (Plotly ScatterMapbox)

Possibilité de sélectionner l’horizon temporel (7j, 30j, 60j, 90j, 180j)

📊 Évaluation

C-index (test) : ~0.80

Mesure la capacité du modèle à bien classer les zones par risque relatif.

☁️ Déploiement & MLOps
🔹 MLflow + NeonDB

Les runs, métriques et versions de modèles sont gérés par MLflow.

NeonDB (PostgreSQL managé) sert de backend store pour les métadonnées MLflow (expériences, runs, registry).

🔹 S3 (Artifacts)

Les modèles entraînés sont stockés dans un bucket S3 (fireprojectbislead).

Exemple :

mlflow/models/xgboost_survivalCOX_model_<run_id>.joblib

🔹 Streamlit App

Application déployée sur Hugging Face Spaces avec Streamlit.

L’app charge directement le pipeline de survie depuis S3 (via boto3 + joblib), sans réentraîner le modèle.

L’utilisateur peut explorer :

Les données

Les résultats du modèle

La carte interactive du risque d’incendie

👉 Lien vers l’application Streamlit
https://huggingface.co/spaces/gdleds/Fire_Projet

🛠️ Améliorations prévues

Ajout de données topographiques (pente, altitude, type de végétation)

Raffinement du modèle (feature engineering avancé, tuning)

Automatisation CI/CD (Airflow, Jenkins, MLflow)

👤 Auteur

Marc Barthes

Développé avec Python, Scikit-learn, XGBoost, Lifelines, Plotly, MLflow, Streamlit
