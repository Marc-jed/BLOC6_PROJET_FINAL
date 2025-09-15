#-------------------------------------------------------- Imports nécessaires ---------------------------------------------------
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.io as pio
import sklearn
import warnings
from scipy.special import expit, logit
import sksurv.datasets
import numpy as np
import joblib
import streamlit as st
import os
from sklearn.cluster import DBSCAN
import urllib.request
import json
import matplotlib
import plotly.graph_objects as go
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from xgboost import DMatrix
from xgboost import train
from lifelines import CoxPHFitter
from itertools import product
from tqdm import tqdm
from xgbse import XGBSEKaplanNeighbors
from xgbse.converters import convert_to_structured
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning
from sklearn import set_config
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ParameterGrid
from sksurv.datasets import load_breast_cancer
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from sksurv.util import Surv
from xgboost import XGBRegressor, DMatrix, train as xgb_train
from lifelines import CoxPHFitter
from mlflow import sklearn as mlflow_sklearn
import mlflow.sklearn
import boto3
import io

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
set_config(display="text")


from sksurv.ensemble import GradientBoostingSurvivalAnalysis


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
set_config(display="text")


#_________________________________________________# Configuration de la page_______________________________________________________
st.set_page_config(page_title="Projet Incendies", layout="wide")

#_________________________________________________# Sidebar de navigation_________________________________________________
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", [
    "Accueil",
    "Notre Projet",
    "Exploration des données",
    "Résultats des modèles",
    
])
#________________________________________________________# Footer#_____________________________________________________________
def show_footer():
    st.markdown("---")
    st.markdown("Projet réalisé dans le cadre de la formation Data Scientist. © 2025")
#_________________________________________________# Chargement DATASET (modèle)#_______________________________________________
@st.cache_data
def load_model_data():
    url = "https://fireprojectbislead.s3.us-east-1.amazonaws.com/dataset/dataset_modele_decompte2.csv"
    try:
        df = pd.read_csv(url)
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données : {e}")
        return pd.DataFrame()
#_________________________________________________# Chargement des données d'incendies et de coordonnées#_______________________________________
@st.cache_data
def load_data():
    url_incendies = 'https://fireprojectbislead.s3.us-east-1.amazonaws.com/dataset/Incendies_2006_2024+(1).csv'
    return pd.read_csv(url_incendies, sep=';', encoding='utf-8', skiprows=3)

@st.cache_data
def load_coords():
    url_coords = 'https://fireprojectbislead.s3.us-east-1.amazonaws.com/dataset/coordonnees_villes+(2).csv'
    return pd.read_csv(url_coords, sep=',', encoding='utf-8')

@st.cache_data
def load_df_merge():
    url = 'https://fireprojectbislead.s3.us-east-1.amazonaws.com/dataset/historique_incendies_avec_coordonnees.csv'
    return pd.read_csv(url, sep=';', encoding='utf-8')
#------------------------------------------------------- ----------------Notre produit#_________________________________________________


# ────────────────────────────────────────────────
# 1) FONCTION DE CHARGEMENT DU CSV BRUT
# ────────────────────────────────────────────────
@st.cache_data(show_spinner="🔄 Téléchargement du CSV…", ttl=None)
def load_raw_data() -> pd.DataFrame:
    url = ("https://fireprojectbislead.s3.us-east-1.amazonaws.com/dataset/dataset_modele_decompte2.csv"
    )
    return pd.read_csv(url, sep=";")


os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT")  # S3
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

def load_model_from_s3(bucket: str, key: str):
    s3 = boto3.client("s3")
    buffer = io.BytesIO()
    s3.download_fileobj(bucket, key, buffer)
    buffer.seek(0)
    model = joblib.load(buffer)
    return model

# ────────────────────────────────────────────────
# 2) FONCTION DE PREDICTIONS (sans entraînement)
# ────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Chargement du modèle…", ttl=None)
def train_model_and_predict(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Retourne df_map prêt pour la carte avec les colonnes
       proba_7j, proba_30j, …, proba_180j."""

    # a) Nettoyage
    df = df_raw.copy()
    df = df.rename(columns={"Feu prévu": "event", "décompte": "duration"})
    df["event"] = df["event"].astype(bool)
    df["duration"] = df["duration"].fillna(0)

    # b) Features
    features = [
        "moyenne precipitations mois", "moyenne temperature mois",
        "moyenne evapotranspiration mois", "moyenne vitesse vent année",
        "moyenne vitesse vent mois", "moyenne temperature année",
        "RR", "UM", "ETPMON", "TN", "TX", "Nombre de feu par an",
        "Nombre de feu par mois", "jours_sans_pluie", "jours_TX_sup_30",
        "ETPGRILLE_7j", "compteur jours vers prochain feu",
        "compteur feu log", "Année", "Mois",
        "moyenne precipitations année", "moyenne evapotranspiration année",
    ]
    features = [f for f in features if f in df.columns]

    # c) Chargement du modèle depuis MLflow
    pipe = load_model_from_s3(
    bucket=os.getenv("S3_BUCKET"),
    key="mlflow/models/xgboost_survivalCOX_model_2f26eb52269844a2b01f13974185f102.joblib"  # ton chemin exacts
)

    # d) Estimation baseline hazard avec Cox factice (comme avant)
    y_struct = Surv.from_dataframe("event", "duration", df)
    ev_train, du_train = y_struct["event"], y_struct["duration"]

    df_fake = pd.DataFrame({
        "duration": du_train,
        "event": ev_train,
        "const": 1,
    })
    dmat = DMatrix(df_fake[["const"]])
    dmat.set_float_info("label", df_fake["duration"])
    dmat.set_float_info("label_lower_bound", df_fake["duration"])
    dmat.set_float_info("label_upper_bound", df_fake["duration"])
    dmat.set_float_info("weight", df_fake["event"])

    bst_fake = xgb_train(
        params={
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "learning_rate": 0.1,
            "max_depth": 1,
            "verbosity": 0,
        },
        dtrain=dmat,
        num_boost_round=100,
    )
    log_hr_fake = bst_fake.predict(dmat)

    df_risque = pd.DataFrame({
        "duration": du_train,
        "event": ev_train,
        "log_risque": log_hr_fake + np.random.normal(0, 1e-4, size=len(log_hr_fake)),
    })
    cph = CoxPHFitter()
    cph.fit(df_risque, duration_col="duration", event_col="event", show_progress=False)
    baseline_cumhaz = cph.baseline_cumulative_hazard_

    def S0(t: int) -> float:
        """Survie de base S0(t) = exp(-H0(t))."""
        idx = baseline_cumhaz.index
        if t in idx:
            H0 = baseline_cumhaz.loc[t].values[0]
        else:
            H0 = baseline_cumhaz.loc[idx[idx <= t]].iloc[-1, 0]
        return float(np.exp(-H0))

    # e) Calcul proba sur plusieurs horizons
    horizons = {7: "proba_7j", 30: "proba_30j", 60: "proba_60j",
                90: "proba_90j", 180: "proba_180j"}

    log_hr_all = pipe.predict(df[features])
    HR = np.exp(log_hr_all)

    for t, col in horizons.items():
        df[col] = 1 - (S0(t) ** HR)   # P(event ≤ t)

    df_map = df[["latitude", "longitude", "ville"] + list(horizons.values())].copy()
    return df_map

# ────────────────────────────────────────────────
# 3) AFFICHAGE SUR LA PAGE « Accueil »
# ────────────────────────────────────────────────
if page == "Accueil":
    st.title("Carte du risque d’incendie en Corse")

    df_raw = load_raw_data()
    df_map = train_model_and_predict(df_raw)

    horizons_lbl = {
        "7 jours":  "proba_7j",
        "30 jours": "proba_30j",
        "60 jours": "proba_60j",
        "90 jours": "proba_90j",
        "180 jours": "proba_180j",
    }
    choix = st.radio(
        "Choisis l’horizon temporel :",
        list(horizons_lbl.keys()),
        horizontal=True,
        index=0,
    )
    col_proba = horizons_lbl[choix]

    # Palette dynamique
    vmax = float(df_map[col_proba].max())
    fig = px.scatter_mapbox(
        df_map,
        lat="latitude",
        lon="longitude",
        hover_name="ville",
        hover_data={col_proba: ":.2%"},
        color=col_proba,
        color_continuous_scale="YlOrRd",  # jaune → orange → rouge
        range_color=(0.0, vmax),
        zoom=7,
        height=650,
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="Probabilité", tickformat=".0%"),
    )

    st.subheader(f"Risque d’incendie – horizon **{choix}**")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------Carte des casernes de pompiers#______________________________________________
from branca.element import Template, MacroElement  # Import nécessaire

import folium
from folium import DivIcon
from streamlit_folium import st_folium

if page == "Accueil":

    # Chargement des données des casernes
    df_casernes = pd.read_csv(
        'https://fireprojectbislead.s3.us-east-1.amazonaws.com/dataset/casernes_corses.csv',
        sep=',',
        encoding='utf8'
    )

    # Nettoyage des coordonnées
    df_casernes['latitude'] = df_casernes['latitude'].astype(str).str.replace(',', '.').astype(float)
    df_casernes['longitude'] = df_casernes['longitude'].astype(str).str.replace(',', '.').astype(float)
    df_casernes = df_casernes.dropna(subset=['latitude', 'longitude'])

    # Catégorisation des casernes
    df_casernes['categorie'] = np.select(
        [
            df_casernes['nom'].str.contains('centre', case=False, na=False),
            df_casernes['nom'].str.contains('base', case=False, na=False),
            df_casernes['nom'].str.contains('SSLIA', case=False, na=False),
            df_casernes['nom'].str.contains('citerne', case=False, na=False),
            df_casernes['nom'].str.contains('borne', case=False, na=False),
        ],
        [
            "Centre d'incendie et de secours",
            'Base forestière',
            'SSLIA (aérodromes)',
            'Citerne',
            'Borne incendie'
        ],
        default='Autre'
    )

    # Dictionnaire d'emojis
    emoji_legende = {
        "Centre d'incendie et de secours": "🚒",
        "Base forestière": "🌲",
        "SSLIA (aérodromes)": "✈️",
        "Citerne": "💦"
    }

    # Carte centrée sur la Corse
    m = folium.Map(location=[42.0396, 9.0129], zoom_start=8)

    for _, row in df_casernes.iterrows():
        emoji = emoji_legende.get(row['categorie'], '❓')
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"{emoji} {row['nom']}",
            icon=DivIcon(html=f"""<div style="font-size:24px">{emoji}</div>""")
        ).add_to(m)

    # Légende HTML
    legend_html = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed; 
        bottom: 50px; left: 50px; width: 280px; 
        background-color: white;
        border: 2px solid grey; 
        z-index: 9999; 
        font-size: 14px;
        color: black;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    ">
    <b>📘 Légende</b><br>
    🚒 Centre d'incendie et de secours<br>
    🌲 Base forestière<br>
    ✈️ SSLIA (aérodromes)<br>
    💦 Citerne<br>
    </div>
    {% endmacro %}
    """

    legend = MacroElement()
    legend._template = Template(legend_html)

    m.get_root().add_child(legend)

    st.subheader("🗺️ Carte des casernes et équipements de lutte contre les incendies")
    st_folium(m, width=1000, height=800)    
#----------------------------------------------------------------------Page Notre Projet---------------------------------------------------
if page == "Notre Projet":
    st.title("🔥 Projet Analyse des Incendies 🔥")

    st.subheader(" 📊 Contexte")
    st.subheader("🌲La forêt française en chiffres")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
La France est le 4ᵉ pays européen en superficie forestière, avec **17,5 millions d’hectares** en métropole (32 % du territoire) et **8 millions** en Guyane.
Au total, les forêts couvrent environ **41 %** du territoire national.

- **75 %** des forêts sont privées (3,5 millions de propriétaires).
- **16 %** publiques (collectivités).
- **9 %** domaniales (État).

La forêt française est un réservoir de biodiversité :  
- **190 espèces d’arbres** (67 % feuillus, 33 % conifères).  
- **73 espèces de mammifères**, **120 d’oiseaux**.  
- Environ **30 000 espèces** de champignons et autant d’insectes.  
- **72 %** de la flore française se trouve en forêt.

Les forêts françaises absorbent environ **9 %** des émissions nationales de gaz à effet de serre, jouant un rôle crucial dans la lutte contre le changement climatique.

Le Code forestier encadre leur gestion durable pour protéger la biodiversité, l’air, l’eau et prévenir les risques naturels.
          """)

if page == "Notre Projet":
    st.header("🔥 Corse : Bilan Campagne Feux de Forêts 2024")

    # Tabs par grande section
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📌 Contexte", "🛠️ Prévention", "🚒 Moyens", "📊 Statistiques",
        "🔍 Causes", "🔎 Enquêtes"
    ])

    with tab1:
        with st.expander("📌 Contexte général"):
            st.markdown("""
- **80 %** de la Corse est couverte de forêts/maquis → **fort risque incendie**  
- **2023-2024** : la plus chaude et la plus sèche jamais enregistrée  
- **714 mm** de pluie sur l’année (**78 %** de la normale)  
- **Façade orientale** : seulement **30 %** des précipitations normales
            """)

    with tab2:
        with st.expander("🛠️ Prévention & Investissements"):
            st.markdown("""
- **1,9 million €** investis en 2023-2024 par l’État (jusqu’à 80 % de financement)  
- Travaux financés :  
  - Pistes DFCI/DECI (Sorio di Tenda, Oletta, Île-Rousse…)  
  - Citernes souples & points d’eau  
  - Drones, caméras thermiques, logiciels SIG  
  - Véhicules pour réserves communales
            """)

    with tab3:
        with st.expander("🚒 Moyens déployés"):
            st.markdown("""
- Jusqu’à **500 personnels mobilisables**  
- **168 sapeurs-pompiers SIS2B**, **261 UIISC5**, forestiers-sapeurs, gendarmerie, ONF…  
- Moyens aériens :  
  - **1 hélico**, **2 canadairs** à Ajaccio  
  - **12 canadairs** + **8 Dashs** nationaux en renfort
            """)

    with tab4:
        with st.expander("📊 Statistiques Feux Été 2024"):
            st.markdown("""
- **107 feux** recensés (~9/semaine)  
- **130 ha** brûlés dont :  
  - 83 % des feux <1 ha : **5,42 ha**  
  - 4 gros feux >10 ha : **72,84 ha**  
  - Linguizetta (**22,19 ha**), Oletta (**18,9 ha**), Pioggiola (**18,75 ha**), Tallone (**13 ha**)  
- Depuis janvier 2024 : **285 feux** pour **587 ha**  
- Feu majeur à Barbaggio : **195 ha** (33 % du total annuel)
            """)

    with tab5:
        with st.expander("🔍 Causes des feux (38 cas identifiés)"):
            st.markdown("""
- **11** : foudre  
- **8** : écobuages  
- **6** : malveillance  
- **5** : accidents  
- **4** : mégots de cigarette  
            """)

        with st.expander("⚠️ Prévention = priorité absolue"):
            st.markdown("""
- **90 %** des feux ont une origine humaine  
- Causes principales : **imprudences** (mégots, BBQ, travaux, écobuages…)
            """)

    with tab6:
        with st.expander("🔎 Enquêtes & Surveillance"):
            st.markdown("""
- **20 incendies** étudiés par la Cellule Technique d’Investigation (CTIFF)  
- Équipes mobilisées : **7 forestiers**, **15 pompiers**, **21 forces de l’ordre**  
- **Fermeture de massif** enclenchée 1 seule fois : forêt de Pinia
            """)
# #---------------------------------------------------Equipe du projet---------------------------------------------------
#     st.subheader("👨‍💻 Équipe du projet")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.image("images/Faycal_Belambri.jpg", width=150)
#         st.markdown("**Fayçal Belambri**\n\nData Scientist\n\nSpécialiste App Streamlit et visualisation")
#     with col2:
#         st.image("images/Joel_Termondjian.jpg", width=150)  
#         st.markdown("**Joël Termondjian**\n\nData Scientist\n\nResponsable des données\n\nPreprocessing\n\nData Enagineering")
#     with col3:
#         st.image("images/Marc_Barthes.jpg", width=150)
#         st.markdown("**Marc Barthes**\n\nData Scientist\n\nML Engineer\n\nExpert en modèles de prédiction")
#---------------------------------------------------Notre Objectif --------------------------------------------------------
  
    st.subheader("🎯 Notre Objectif")
    st.markdown("""
Dans un contexte de **changement climatique** et de **risques accrus d’incendies de forêt**, notre équipe a développé un projet innovant visant à **analyser et prédire les zones à risque d’incendie** en France, avec un focus particulier sur la **Corse**.
    """)
#---------------------------------------------------Obectifs du projet---------------------------------------------------
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🔍 Exploration des données")
        st.markdown("""
- ✅ **Évolution du nombre d’incendies**, répartition par mois et par causes.
- ✅ **Cartographie interactive** des incendies sur tout le territoire.
- ✅ **Analyse des clusters** grâce à DBSCAN pour identifier les zones les plus à risque.
        """)

    with col2:
        st.subheader("📈 Modèles prédictifs")
        st.markdown("""
- ✅ **Comparaison des modèles** : Random Forest, XGBoost, analyse de survie.
- ✅ **Prédiction des zones à risque** avec visualisation sur carte.
- ✅ Fourniture d'un **outil décisionnel** pour les autorités et les services de gestion des risques.
        """)

    st.subheader("📘 Définition de l'analyse de survie (Survival Analysis")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 🧠 Qu’est-ce que l’analyse de survie ?")
        st.markdown("""
L’**analyse de survie** (ou **Survival Analysis**) est une méthode statistique utilisée pour **modéliser le temps avant qu’un événement se produise**, comme :
- 🔥 un incendie,
- 🏥 un décès,
- 📉 une résiliation d’abonnement,
- 🧯 une panne.
""")

    with col2:
        st.markdown("### 📌 Objectif :")
        st.markdown("""
> Estimer la **probabilité qu’un événement ne se soit pas encore produit** à un instant donné.
""")

    with col3:
        st.markdown("### 🔑 Concepts fondamentaux : ")
        st.markdown("""
- ⏳ **Temps de survie (`T`)** : temps écoulé jusqu’à l’événement.
- 🎯 **Événement** : le phénomène qu’on cherche à prédire (feu, panne, décès...).
- ❓ **Censure** : l’événement **n’a pas encore eu lieu** durant la période d’observation.
- 📉 **Fonction de survie `S(t)`** : probabilité de "survivre" après le temps `t`.
- ⚠️ **Fonction de risque `h(t)`** : probabilité que l’événement se produise **immédiatement après `t`**, sachant qu’il ne s’est pas encore produit.
""")
    
    with col4:
        st.markdown ("### 🧪 Exemples d’applications :")
        st.markdown("""
| Domaine | Exemple |
|--------|---------|
| 🔥 Incendies | Quand un feu va-t-il se déclarer ? |
| 🏥 Santé | Combien de temps un patient survivra après traitement ? |
| 📉 Marketing | Quand un client risque-t-il de partir ? |
| 🧑‍💼 RH | Quand un salarié quittera-t-il l’entreprise ? |

""")

    show_footer()

#---------------------------------------------------# Page EDA  -----------------------------------------------------------------

if page == "Exploration des données":
    st.title("🗺️ Visualisation des incendies entre 2006 et 2024")

    df = load_data()
    coords = load_coords()
    df_merge = load_df_merge()

    st.subheader("Aperçu des coordonnées des villes")

    fig = px.scatter_map(
        coords, 
        lat="latitude", 
        lon="longitude", 
        hover_name="ville",
        height=800,
        zoom=5,
        map_style="carto-positron",
        title="Carte interactive des communes (coordonnées)"
    )
    st.plotly_chart(fig, use_container_width=True)

#---------------------------------------------------# DBSCAN Clustering---------------------------------------------------
    st.subheader("🔥 Détection des clusters d'incendies avec DBSCAN")

    commune_counts = df_merge.groupby(['Nom de la commune', 'latitude', 'longitude']).size().reset_index(name='frequence')
    df_expanded = commune_counts.loc[commune_counts.index.repeat(commune_counts['frequence'])].reset_index(drop=True)

    X = df_expanded[['latitude', 'longitude']]
    coords_rad = np.radians(X)
    kms_per_radian = 6371.0088
    eps_km = 5
    eps = eps_km / kms_per_radian

    db = DBSCAN(eps=eps, min_samples=20, metric='haversine').fit(coords_rad)
    df_expanded['cluster'] = db.labels_

    clustered_data = df_expanded[df_expanded['cluster'] != -1]

    fig = px.scatter_map(
        clustered_data,
        lat="latitude",
        lon="longitude",
        color="cluster",
        hover_name="Nom de la commune",
        zoom=5,
        height=900,
        title="🔥 Clusters d'incendies en France (2006-2024) détectés par DBSCAN",
        map_style="carto-positron"
    )
    st.plotly_chart(fig, use_container_width=True)


    #---------------------------------------------------Histogramme mensuel#---------------------------------------------------
    st.title("Comparaison mensuelle des incendies par année")

    df_temp = df_merge.copy()
    df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce')
    df_temp = df_temp.dropna(subset=['Date'])

    df_temp['mois'] = df_temp['Date'].dt.month
    df_temp['année'] = df_temp['Date'].dt.year

    mois_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df_temp['mois_nom'] = df_temp['mois'].apply(lambda x: mois_abbr[x - 1])
    df_temp['mois_nom'] = pd.Categorical(df_temp['mois_nom'], categories=mois_abbr, ordered=True)

    df_grouped = df_temp.groupby(['mois_nom', 'année']).size().reset_index(name='nombre_feux')

    fig = px.bar(
        df_grouped,
        x='mois_nom',
        y='nombre_feux',
        color='année',
        barmode='group',
        title='Comparaison mensuelle des incendies par année',
        height=600,
        width=1000
    )
    st.plotly_chart(fig, use_container_width=True)

    show_footer()

    #--------------------------------------------------- Analyse des causes---------------------------------------------------
    causes = df_merge['Nature'].value_counts()
    st.subheader("Répartition des causes d'incendies")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(
        causes.values,
        labels=causes.index,
        autopct='%1.1f%%',
        startangle=140,
        shadow=True,
        explode=[0.05]*len(causes)
    )
    ax.set_title("Répartition des causes d'incendies")
    ax.axis('equal')
    st.pyplot(fig)

#---------------------------------------------------- Nombre total d’incendies par année-----------------------------------------

if page == "Exploration des données":
    st.title("Analyse des incendies par année 🔥")

    # Copie du DataFrame
    df_temp = df_merge.copy()

    # Conversion de la colonne Date
    df_temp['Date'] = pd.to_datetime(df_temp['Date'])
    df_temp['année'] = df_temp['Date'].dt.year

    # Regroupement par année uniquement
    df_grouped = df_temp.groupby('année').size().reset_index(name='nombre_feux')

    # Création du graphique en barres
    fig = px.bar(
        df_grouped,
        x='année',
        y='nombre_feux',
        title='Nombre total d’incendies par année',
        height=600,
        width=1200,
        text='nombre_feux'
    )

    fig.update_xaxes(
        tickmode='linear',
        dtick=1  # une année à chaque tick
    )

    fig.update_layout(
        xaxis_title='Année',
        yaxis_title='Nombre de feux',
        xaxis_tickangle=0
    )

    st.plotly_chart(fig)
#---------------------------------------------------- Page Exploration des données -----------------------------------------

if page == "Exploration des données":

#---------------------------------------------------- Les 10 départements avec le plus d’incendies -----------------------------------------

    # Copie du DataFrame
    df_temp = df_merge.copy()

    # Regroupement par département
    df_grouped = df_temp.groupby('Département').size().reset_index(name='nombre_feux')

    # Classement décroissant et sélection du top 10
    df_top10 = df_grouped.sort_values(by='nombre_feux', ascending=False).head(10)

    # Graphique en barres
    fig = px.bar(
        df_top10,
        x='Département',
        y='nombre_feux',
        title='Les 10 départements avec le plus d’incendies',
        height=600,
        width=1000,
        text='nombre_feux'
    )

    # Fond clair
    fig.update_layout(
        template='plotly_white',
        xaxis_title='Département',
        yaxis_title='Nombre de feux',
        xaxis_tickangle=-45,
    )

    # Texte au-dessus des barres
    fig.update_traces(textposition='outside')

    # Affichage dans l'app
    st.plotly_chart(fig)


 #---------------------------------------------------- Les 10 départements les plus touchés -----------------------------------------   

if page == "Exploration des données":

    # 🔎 Vérification rapide du DataFrame
    if "Département" not in df_merge.columns:
        st.error("❌ La colonne 'Département' est absente du DataFrame.")
    elif df_merge.empty:
        st.warning("⚠️ Le DataFrame est vide.")
    else:
        # ✅ Copie et nettoyage du DataFrame
        df_temp = df_merge.copy()
        df_temp = df_temp[df_temp['Département'].notna()]  # Supprime les lignes sans département

        # 📊 Regroupement par département
        df_grouped = df_temp.groupby('Département').size().reset_index(name='nombre_feux')

        # 🔢 Total général
        total_feux = df_grouped['nombre_feux'].sum()

        # 🔝 Top 10 des départements
        df_top10 = df_grouped.sort_values(by='nombre_feux', ascending=False).head(10)

        # 📈 Calcul des proportions
        df_top10['proportion_totale'] = df_top10['nombre_feux'] / total_feux

        # 🥧 Création du graphique circulaire
        fig_pie = px.pie(
            df_top10,
            names='Département',
            values='nombre_feux',
            title='Les 10 départements les plus touchés (proportion sur le total global)',
        )
        fig_pie.update_traces(textinfo='label+percent')

        # 📌 Affichage dans l'app
        st.plotly_chart(fig_pie)

#---------------------------------------------------- Carte des feux par département (2006-2024) ---------------------------------------------------------

if page == "Exploration des données":
    st.subheader("Carte des feux par département (2006-2024)")

    # Copie du dataset
    df_temp = df_merge.copy()

    # Codes départements formatés
    df_temp['Département'] = df_temp['Département'].astype(str).str.zfill(2)
    df_grouped = df_temp.groupby('Département').size().reset_index(name='nombre_feux')

    # Chargement GeoJSON
    url_geojson = 'https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson'
    with urllib.request.urlopen(url_geojson) as response:
        departements_geojson = json.load(response)

    # Carte choroplèthe
    fig = px.choropleth(
        df_grouped,
        geojson=departements_geojson,
        locations='Département',
        featureidkey='properties.code',
        color='nombre_feux',
        color_continuous_scale='OrRd',
        title='Total des feux par département (2006-2024)',
        labels={'nombre_feux': 'Feux'},
    )

    # Style géographique
    fig.update_geos(
        visible=False,
        lataxis_range=[41, 52],
        lonaxis_range=[-5.5, 10],
        showcountries=False,
        showcoastlines=False,
        showland=True,
        landcolor='white',
        fitbounds="locations"
    )

    # Mise en page
    fig.update_layout(
        template='plotly_white',
        width=1000,
        height=700,
        margin=dict(l=0, r=20, t=40, b=0),
        coloraxis_colorbar=dict(
            title="Feux",
            thickness=15,
            len=0.4,
            y=0.5
        )
    )

    # Affichage Streamlit
    st.plotly_chart(fig)
#---------------------------------------------------- Les 10 départements avec le plus d’incendies -----------------------------------------
# import plotly.graph_objects as go

# -----------------------------------------------------------
# 🔥 Top-10 des départements par nombre de feux – version GO
# -----------------------------------------------------------
if page == "Exploration des données":
    df_temp = df_merge.copy()
    df_temp["Département"] = df_temp["Département"].replace(
        {"2A": "2A/2B", "2B": "2A/2B"}
    )

    df_count = (
        df_temp.groupby("Département")
        .size()
        .reset_index(name="Nombre de feux")
        .sort_values("Nombre de feux", ascending=False)
        .head(10)
    )

    # ── Barres avec labels (Graph Objects)
    fig_top10_feux = go.Figure(
        data=go.Bar(
            x=df_count["Département"],
            y=df_count["Nombre de feux"],
            text=df_count["Nombre de feux"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(size=16, color="#2e2e2e"),  # police foncée
            marker=dict(
                color="#627CFF",
                line=dict(color="black", width=1.5),
            ),
        )
    )

    # ── Mise en page inspirée de ta Fig 1
    fig_top10_feux.update_layout(
        title="🔥 Top 10 des départements avec le plus d’incendies",
        title_font_size=28,
        template="plotly_white",                     # fond clair + grille
        plot_bgcolor="rgba(245,248,255,1)",
        paper_bgcolor="rgba(245,248,255,1)",
        margin=dict(l=80, r=80, t=110, b=120),
        font=dict(size=18, color="#2e2e2e"),        # police par défaut foncée
        xaxis=dict(
            title="Département",
            tickangle=-35,
            tickfont=dict(size=16),
        ),
        yaxis=dict(
            title="Nombre de feux",
            tickformat=",d",
            tickfont=dict(size=16),
        ),
        bargap=0.05,
    )

    st.plotly_chart(fig_top10_feux, use_container_width=True)


#---------------------------------------------------- Page Résultats des modèles -----------------------------------------

elif page == "Résultats des modèles":
    st.title("📈 Résultats des modèles prédictifs")
    st.markdown("### Comparaison des modèles de Survival Analysis")

    #--------------------------------------------------- Tableau codé en dur en Markdown -----------------------------------
    st.markdown("""
    | Modèle                            | Concordance Index | 
    |-----------------------------------|-------------------|
    | Predict survival fonction (MVP)   | 0.691             |                 
    | XGBOOST survival cox              | 0.809             |      
    """)

    st.markdown("👉 Le modèle **XGBOOST survival cox** obtient la meilleure performance globale.")

    show_footer()