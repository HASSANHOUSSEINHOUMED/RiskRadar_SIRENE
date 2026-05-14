"""
RiskRadar - Dashboard de prédiction du risque de cessation d'activité des entreprises françaises
Application Streamlit destinée aux utilisateurs non-techniques
Source des données : INSEE SIRENE / data.gouv.fr
"""

import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
import plotly.graph_objects as go
import joblib
import os
import json
import time
import random
import math
from datetime import datetime
from huggingface_hub import hf_hub_download

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
st.set_page_config(
    page_title="RiskRadar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif !important; }
    .stApp { background-color: #f8fafc; }
    section[data-testid="stSidebar"] { display: none; }
    #MainMenu, footer, header { visibility: hidden; }
    div[data-testid="metric-container"] {
        background: white;
        border: 0.5px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    div[data-testid="metric-container"] label {
        font-size: 0.6rem !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    .stButton button {
        background: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 0.75rem 1.5rem !important;
        width: 100% !important;
    }
    .stButton button:hover {
        background: #1d4ed8 !important;
        box-shadow: 0 4px 15px rgba(37,99,235,0.3) !important;
    }
    .stTextInput input {
        background: white !important;
        border: 1.5px solid #cbd5e1 !important;
        color: #0f172a !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1rem !important;
    }
    .stTextInput input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
    }
    .stTextInput input::placeholder { color: #94a3b8 !important; }
    [data-testid="stStatusWidget"] { display: none !important; }
    [data-testid="stCaptionContainer"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# CONSTANTES
# -------------------------------------------------------
DATE_REFERENCE = pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))
CHEMIN_SILVER = "data/silver/sirene_entreprises_silver_latest.parquet"
CHEMIN_PERFORMANCES = "data/modeles/performances.json"
DOSSIER_MODELES = "data/modeles"

# Identifiant du dataset Hugging Face
HF_REPO_ID = "HassanHH2910/riskradar-sirene"

ORDRE_TRANCHES = {
    "NN": 0, "00": 1, "01": 2, "02": 3, "03": 4,
    "11": 5, "12": 6, "21": 7, "22": 8, "31": 9,
    "32": 10, "41": 11, "42": 12, "51": 13, "52": 14, "53": 15
}
ORDRE_CATEGORIE = {"TPE": 1, "PME": 2, "ETI": 3, "GE": 4}

LABEL_TRANCHES = {
    "NN": "Non renseigné", "00": "0 salarié", "01": "1 à 2 salariés",
    "02": "3 à 5 salariés", "03": "6 à 9 salariés", "11": "10 à 19 salariés",
    "12": "20 à 49 salariés", "21": "50 à 99 salariés", "22": "100 à 199 salariés",
    "31": "200 à 249 salariés", "32": "250 à 499 salariés", "41": "500 à 999 salariés",
    "42": "1 000 à 1 999 salariés", "51": "2 000 à 4 999 salariés",
    "52": "5 000 à 9 999 salariés", "53": "10 000 salariés et plus"
}

LABEL_CATEGORIE = {
    "TPE": "Très petite entreprise",
    "PME": "Petite et moyenne entreprise",
    "ETI": "Entreprise de taille intermédiaire",
    "GE": "Grande entreprise"
}

NAF_NOMS = {
    "68.20B": "Location immobilière",
    "68.20A": "Location de logements",
    "70.22Z": "Conseil aux entreprises",
    "94.99Z": "Associations diverses",
    "47.11B": "Commerce alimentaire",
    "41.20A": "Construction maisons",
    "56.10A": "Restauration traditionnelle",
    "62.01Z": "Programmation informatique",
    "43.21A": "Électricité bâtiment",
    "45.20A": "Réparation automobile",
    "96.02A": "Coiffure",
    "47.71Z": "Commerce habillement",
    "86.21Z": "Médecine générale",
    "41.10A": "Promotion immobilière",
    "64.19Z": "Banque et crédit",
    "65.12Z": "Assurances",
    "62.02A": "Conseil informatique",
    "56.10C": "Restauration rapide",
    "43.22A": "Plomberie chauffage",
    "47.25Z": "Commerce boissons",
    "85.59A": "Formation continue",
    "86.22C": "Spécialités médicales",
    "49.41A": "Transport routier",
    "43.31Z": "Plâtrerie",
    "41.20B": "Construction bâtiments",
    "47.19B": "Commerce divers",
    "81.10Z": "Services bâtiments",
    "64.20Z": "Holdings financières",
    "77.11A": "Location voitures",
    "82.11Z": "Services administratifs",
    "74.10Z": "Design",
    "90.01Z": "Arts du spectacle",
    "47.91A": "Commerce en ligne",
    "55.10Z": "Hôtellerie",
    "73.11Z": "Publicité",
    "43.12A": "Terrassement",
    "45.11Z": "Commerce automobiles"
}


def naf_lisible(code):
    code_str = str(code).strip()
    return NAF_NOMS.get(code_str, code_str)


def formater_kpi(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)


def formater_nombre(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.0f}K"
    return f"{n:,}"


def formater_date(date_str):
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        mois_fr = [
            "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
            "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
        ]
        return f"{mois_fr[date_obj.month - 1]} {date_obj.year}"
    except Exception:
        return date_str


def etoiles(auc_roc):
    if auc_roc >= 98:
        return "★★★★★"
    elif auc_roc >= 90:
        return "★★★★☆"
    elif auc_roc >= 80:
        return "★★★☆☆"
    else:
        return "★★☆☆☆"


# -------------------------------------------------------
# TÉLÉCHARGEMENT DEPUIS HUGGING FACE
# Télécharge les fichiers uniquement s'ils ne sont pas déjà présents
# -------------------------------------------------------
def telecharger_depuis_hf():
    os.makedirs("data/silver", exist_ok=True)
    os.makedirs(DOSSIER_MODELES, exist_ok=True)

    fichiers = [
        ("sirene_entreprises_silver_latest.parquet", CHEMIN_SILVER),
        ("performances.json", CHEMIN_PERFORMANCES),
        ("modele_xgboost.pkl", os.path.join(DOSSIER_MODELES, "modele_xgboost.pkl")),
        ("modele_random_forest.pkl", os.path.join(DOSSIER_MODELES, "modele_random_forest.pkl")),
        ("modele_logistic_regression.pkl", os.path.join(DOSSIER_MODELES, "modele_logistic_regression.pkl")),
    ]

    for nom_hf, chemin_local in fichiers:
        if not os.path.exists(chemin_local):
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=nom_hf,
                repo_type="dataset",
                local_dir=os.path.dirname(chemin_local),
                local_dir_use_symlinks=False
            )


# Téléchargement au démarrage — ne fait rien si les fichiers existent déjà
telecharger_depuis_hf()

# -------------------------------------------------------
# CHARGEMENT DES DONNÉES
# -------------------------------------------------------
@st.cache_data
def charger_performances():
    if not os.path.exists(CHEMIN_PERFORMANCES):
        return None
    with open(CHEMIN_PERFORMANCES, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def charger_statistiques_globales():
    # Lecture uniquement des colonnes nécessaires pour éviter de charger 29M lignes en RAM
    colonnes = [
        "etatAdministratifUniteLegale",
        "activitePrincipaleUniteLegale",
        "categorieEntreprise"
    ]
    df = pq.read_table(CHEMIN_SILVER, columns=colonnes).to_pandas()
    nb = pq.read_metadata(CHEMIN_SILVER).num_rows
    nb_actives = int((df["etatAdministratifUniteLegale"] == "A").sum())
    nb_cessees = int((df["etatAdministratifUniteLegale"] == "C").sum())
    pct_actives = round(nb_actives / nb * 100, 1)
    pct_cessees = round(nb_cessees / nb * 100, 1)
    top_secteurs = df["activitePrincipaleUniteLegale"].dropna().value_counts().head(4)
    repartition_cat = df["categorieEntreprise"].value_counts(dropna=True)
    nb_non_classes = int(df["categorieEntreprise"].isna().sum())
    return nb, nb_actives, nb_cessees, pct_actives, pct_cessees, top_secteurs, repartition_cat, nb_non_classes


@st.cache_data
def chercher_entreprise_parquet(siren):
    # Recherche d'une seule ligne par SIREN sans charger les 29M lignes en RAM
    siren_padded = str(siren).zfill(9)
    table = pq.read_table(CHEMIN_SILVER)
    masque = pc.equal(pc.cast(table["siren"], "string"), siren_padded)
    resultat = table.filter(masque).to_pandas()
    if len(resultat) == 0:
        return None
    return resultat.iloc[0]


@st.cache_resource
def charger_modele(chemin):
    if not os.path.exists(chemin):
        return None
    return joblib.load(chemin)


# -------------------------------------------------------
# FONCTIONS MÉTIER
# -------------------------------------------------------
def preparer_features(entreprise):
    # Reproduction exacte des transformations appliquées lors de l'étape Gold
    features = {}
    features["categorieJuridiqueUniteLegale"] = 0.01
    features["activitePrincipaleUniteLegale"] = 0.01
    tranche = str(entreprise.get("trancheEffectifsUniteLegale", "NN"))
    features["trancheEffectifsUniteLegale"] = ORDRE_TRANCHES.get(tranche, 0)
    categorie = str(entreprise.get("categorieEntreprise", ""))
    features["categorieEntreprise"] = ORDRE_CATEGORIE.get(categorie, 0)
    ess = str(entreprise.get("economieSocialeSolidaireUniteLegale", ""))
    features["economieSocialeSolidaireUniteLegale"] = 1 if ess == "O" else (0 if ess == "N" else -1)
    nb_periodes = entreprise.get("nombrePeriodesUniteLegale", 1)
    try:
        features["nombrePeriodesUniteLegale"] = min(float(nb_periodes) / 50.0, 1.0)
    except Exception:
        features["nombrePeriodesUniteLegale"] = 0.0
    try:
        date_creation = pd.to_datetime(entreprise.get("dateCreationUniteLegale"), errors="coerce")
        if pd.isna(date_creation):
            features["anciennete_annees"] = 0.0
        else:
            anciennete = (DATE_REFERENCE - date_creation).days / 365.25
            features["anciennete_annees"] = min(max(anciennete, 0) / 100.0, 1.0)
    except Exception:
        features["anciennete_annees"] = 0.0
    try:
        date_debut = pd.to_datetime(entreprise.get("dateDebut"), errors="coerce")
        if pd.isna(date_debut):
            features["duree_periode_actuelle_annees"] = 0.0
        else:
            duree = (DATE_REFERENCE - date_debut).days / 365.25
            features["duree_periode_actuelle_annees"] = min(max(duree, 0) / 50.0, 1.0)
    except Exception:
        features["duree_periode_actuelle_annees"] = 0.0
    colonnes = [
        "categorieJuridiqueUniteLegale", "activitePrincipaleUniteLegale",
        "trancheEffectifsUniteLegale", "categorieEntreprise",
        "economieSocialeSolidaireUniteLegale", "nombrePeriodesUniteLegale",
        "anciennete_annees", "duree_periode_actuelle_annees"
    ]
    return pd.DataFrame([features])[colonnes]


def calculer_anciennete(entreprise):
    try:
        date_creation = pd.to_datetime(entreprise.get("dateCreationUniteLegale"), errors="coerce")
        if pd.isna(date_creation):
            return "Non renseignée", None
        anciennete = int((DATE_REFERENCE - date_creation).days / 365.25)
        return f"{anciennete} an{'s' if anciennete > 1 else ''}", date_creation.year
    except Exception:
        return "Non renseignée", None


def obtenir_verdict(proba):
    score = proba * 100
    if proba < 0.35:
        return (
            "✅", "Faible risque de fermeture future",
            "D'après mon analyse, cette entreprise a peu de chances de fermer dans un avenir proche. "
            "Son profil correspond aux caractéristiques des entreprises qui perdurent dans le temps.",
            "#16a34a", "#f0fdf4", "#86efac", score, "Stable"
        )
    elif proba < 0.65:
        return (
            "⚠️", "Quelques signaux à surveiller",
            "Mon système détecte des caractéristiques mixtes. Certains indicateurs sont rassurants, "
            "d'autres méritent une attention particulière. Un suivi régulier est recommandé.",
            "#d97706", "#fffbeb", "#fcd34d", score, "À surveiller"
        )
    else:
        return (
            "🚨", "Risque élevé de fermeture",
            "Cette entreprise présente des caractéristiques proches de celles des entreprises qui ont "
            "cessé leur activité dans ma base de données. Une analyse approfondie est recommandée.",
            "#dc2626", "#fef2f2", "#fca5a5", score, "Risque élevé"
        )


# -------------------------------------------------------
# GRAPHIQUES PLOTLY
# -------------------------------------------------------
@st.cache_data
def creer_donut(pct_actives, pct_cessees):
    fig = go.Figure(data=[go.Pie(
        labels=[f"Actives {pct_actives}%", f"Fermées {pct_cessees}%"],
        values=[pct_actives, pct_cessees],
        hole=0.65,
        marker_colors=["#16a34a", "#dc2626"],
        textinfo="none",
        hovertemplate="%{label}<extra></extra>"
    )])
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            x=1.05, y=0.5,
            xanchor="left", yanchor="middle",
            font=dict(size=11, color="#64748b"),
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(t=5, b=5, l=5, r=80),
        height=160,
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    return fig


@st.cache_data
def creer_barres_secteurs(codes, valeurs):
    labels = [naf_lisible(c) for c in codes]
    textes = [formater_nombre(int(v)) for v in valeurs]
    fig = go.Figure(go.Bar(
        x=list(valeurs), y=labels,
        orientation="h",
        marker_color="#2563eb", opacity=0.7,
        text=textes,
        textposition="outside",
        textfont=dict(size=9, color="#64748b"),
        hovertemplate="%{y} : %{text}<extra></extra>"
    ))
    fig.update_layout(
        margin=dict(t=5, b=5, l=5, r=60),
        height=160, paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(tickfont=dict(size=10, color="#64748b"), autorange="reversed")
    )
    return fig


@st.cache_data
def creer_barres_tailles(categories, valeurs):
    valeurs_log = [math.log10(v + 1) for v in valeurs]
    textes = [formater_nombre(v) for v in valeurs]
    couleurs = ["#1d4ed8", "#3b82f6", "#93c5fd", "#bfdbfe", "#e2e8f0"]
    fig = go.Figure(go.Bar(
        x=categories, y=valeurs_log,
        marker_color=couleurs[:len(categories)],
        text=textes, textposition="outside",
        textfont=dict(size=9, color="#64748b"),
        hovertemplate="%{x} : %{text}<extra></extra>"
    ))
    fig.update_layout(
        margin=dict(t=20, b=5, l=5, r=5),
        height=160, paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(tickfont=dict(size=9, color="#64748b")),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        showlegend=False
    )
    return fig


# -------------------------------------------------------
# CHARGEMENT DES PERFORMANCES
# -------------------------------------------------------
performances = charger_performances()
if performances is None:
    st.error("Configuration manquante. Veuillez exécuter le script 07_ml_models.py.")
    st.stop()

modeles_data = performances["modeles"]
modele_recommande = next(
    (k for k, v in modeles_data.items() if v.get("recommande")),
    list(modeles_data.keys())[0]
)
date_entrainement = performances.get("date_entrainement", "N/A")
auc_rec = modeles_data[modele_recommande]["auc_roc"]
nb_correct = int(modeles_data[modele_recommande]["accuracy"])
modele_choisi = modele_recommande

# -------------------------------------------------------
# ÉCRAN DE CHARGEMENT
# -------------------------------------------------------
if "pret" not in st.session_state:
    if not os.path.exists(CHEMIN_SILVER):
        st.error("Base de données introuvable. Vérifiez que le pipeline a été exécuté.")
        st.stop()

    taille_mb = int(os.path.getsize(CHEMIN_SILVER) / (1024 * 1024))
    astuces = [
        "J'ai entraîné mon intelligence artificielle sur des millions d'entreprises réelles pour vous offrir une analyse fiable en quelques secondes.",
        "Toutes les entreprises de France sont disponibles — actives et fermées depuis 1973.",
        "Je prédis le risque futur de fermeture — pas seulement l'état actuel, mais ce qui pourrait arriver demain."
    ]
    astuce = random.choice(astuces)

    _, col_splash, _ = st.columns([1, 2, 1])
    with col_splash:
        st.markdown(
            f"""
            <div style="background:white; border:0.5px solid #e2e8f0; border-radius:20px;
                        padding:3rem 2rem; text-align:center; margin:2rem 0;">
                <div style="font-size:3.5rem; margin-bottom:1rem;">📡</div>
                <div style="font-size:1.5rem; font-weight:600; color:#0f172a; margin-bottom:0.5rem;">
                    Bienvenue sur RiskRadar
                </div>
                <div style="font-size:0.88rem; color:#64748b; line-height:1.8; margin-bottom:1.5rem;">
                    Chargement de la base officielle des entreprises françaises.<br>
                    Cela prend quelques secondes — merci de votre patience !<br>
                    <span style="font-size:0.75rem; color:#94a3b8;">Fichier : {taille_mb} MB</span>
                </div>
                <div style="background:#eff6ff; border:0.5px solid #bfdbfe; border-radius:12px;
                            padding:0.9rem 1.2rem; font-size:0.82rem; color:#1e40af; line-height:1.7;">
                    {astuce}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        barre = st.progress(0)
        barre.progress(30, text="Lecture des données en cours...")
        charger_statistiques_globales()
        barre.progress(100, text="Chargement terminé — RiskRadar est prêt !")
        time.sleep(0.8)

    st.session_state.pret = True
    st.rerun()

# -------------------------------------------------------
# DONNÉES GLOBALES
# -------------------------------------------------------
if "stats" not in st.session_state:
    (
        nb_entreprises, nb_actives, nb_cessees,
        pct_actives, pct_cessees,
        top_secteurs, repartition_cat, nb_non_classes
    ) = charger_statistiques_globales()
    st.session_state.stats = {
        "nb_entreprises": nb_entreprises,
        "nb_actives": nb_actives,
        "nb_cessees": nb_cessees,
        "pct_actives": pct_actives,
        "pct_cessees": pct_cessees,
        "top_secteurs": top_secteurs,
        "repartition_cat": repartition_cat,
        "nb_non_classes": nb_non_classes
    }

stats = st.session_state.stats
nb_entreprises = stats["nb_entreprises"]
nb_actives = stats["nb_actives"]
nb_cessees = stats["nb_cessees"]
pct_actives = stats["pct_actives"]
pct_cessees = stats["pct_cessees"]
top_secteurs = stats["top_secteurs"]
repartition_cat = stats["repartition_cat"]
nb_non_classes = stats["nb_non_classes"]

ordre_cat = ["TPE", "PME", "ETI", "GE", "Non classées"]
valeurs_cat = [
    int(repartition_cat.get("TPE", 0)),
    int(repartition_cat.get("PME", 0)),
    int(repartition_cat.get("ETI", 0)),
    int(repartition_cat.get("GE", 0)),
    nb_non_classes
]

date_lisible = formater_date(date_entrainement)

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
col_logo, col_info = st.columns([1, 1])
with col_logo:
    st.markdown(
        """
        <div style="padding:1.2rem 0 0.8rem 0; border-bottom:0.5px solid #e2e8f0; margin-bottom:1.8rem;">
            <div style="font-size:1.6rem; font-weight:700; color:#0f172a; letter-spacing:-0.5px;">
                📡 Risk<span style="color:#2563eb;">Radar</span>
            </div>
            <div style="font-size:0.62rem; color:#94a3b8; letter-spacing:1.5px; text-transform:uppercase;">
                Prédiction du risque de cessation d'activité — France
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_info:
    st.markdown(
        f"""
        <div style="padding:1.2rem 0 0.8rem 0; border-bottom:0.5px solid #e2e8f0; margin-bottom:1.8rem;
                    text-align:right; font-size:0.68rem; color:#94a3b8; line-height:1.9;">
            Mémoire de recherche — Mastère 2 Big Data, IA et Dév — École IPSSI 2026<br>
            Source : INSEE SIRENE / data.gouv.fr — {DATE_REFERENCE.strftime('%d/%m/%Y')}
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------------
# KPIs
# -------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Entreprises dans la base", formater_kpi(nb_entreprises))
k2.metric("Actives aujourd'hui", formater_kpi(nb_actives))
k3.metric("Ont fermé par le passé", formater_kpi(nb_cessees))
k4.metric("Données du mois", date_lisible)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------
# GRAPHIQUES
# -------------------------------------------------------
g1, g2, g3 = st.columns(3)

with g1:
    st.markdown(
        "<div style='background:white; border:0.5px solid #e2e8f0; border-radius:12px; "
        "padding:1rem 1rem 0 1rem;'>"
        "<div style='font-size:0.6rem; color:#2563eb; text-transform:uppercase; "
        "letter-spacing:2px; font-weight:600; margin-bottom:4px;'>Actives vs fermées</div>",
        unsafe_allow_html=True
    )
    st.plotly_chart(
        creer_donut(pct_actives, pct_cessees),
        width="stretch",
        config={"displayModeBar": False}
    )
    st.markdown(
        "<div style='font-size:0.68rem; color:#94a3b8; padding:0 0.8rem 0.8rem 0.8rem; line-height:1.6;'>"
        "Répartition de toutes les entreprises françaises — actives et fermées depuis 1973."
        "</div></div>",
        unsafe_allow_html=True
    )

with g2:
    st.markdown(
        "<div style='background:white; border:0.5px solid #e2e8f0; border-radius:12px; "
        "padding:1rem 1rem 0 1rem;'>"
        "<div style='font-size:0.6rem; color:#2563eb; text-transform:uppercase; "
        "letter-spacing:2px; font-weight:600; margin-bottom:4px;'>Top 4 secteurs principaux</div>",
        unsafe_allow_html=True
    )
    st.plotly_chart(
        creer_barres_secteurs(list(top_secteurs.index), list(top_secteurs.values)),
        width="stretch",
        config={"displayModeBar": False}
    )
    st.markdown(
        "<div style='font-size:0.68rem; color:#94a3b8; padding:0 0.8rem 0.8rem 0.8rem; line-height:1.6;'>"
        "Top 4 secteurs sur des centaines — actives et fermées confondues."
        "</div></div>",
        unsafe_allow_html=True
    )

with g3:
    st.markdown(
        "<div style='background:white; border:0.5px solid #e2e8f0; border-radius:12px; "
        "padding:1rem 1rem 0 1rem;'>"
        "<div style='font-size:0.6rem; color:#2563eb; text-transform:uppercase; "
        "letter-spacing:2px; font-weight:600; margin-bottom:4px;'>Taille des entreprises</div>",
        unsafe_allow_html=True
    )
    st.plotly_chart(
        creer_barres_tailles(ordre_cat, valeurs_cat),
        width="stretch",
        config={"displayModeBar": False}
    )
    st.markdown(
        "<div style='font-size:0.68rem; color:#94a3b8; padding:0 0.8rem 0.8rem 0.8rem; line-height:1.6;'>"
        "\"Non classées\" : entreprises sans catégorie renseignée dans la source INSEE — actives et fermées confondues."
        "</div></div>",
        unsafe_allow_html=True
    )

# -------------------------------------------------------
# BANNIÈRE IA
# -------------------------------------------------------
st.markdown(
    f"""
    <div style="background:#eff6ff; border:0.5px solid #bfdbfe; border-radius:14px;
                padding:1.2rem 1.8rem; margin:1.2rem 0;
                display:flex; align-items:center; gap:1.5rem; flex-wrap:wrap;">
        <div style="font-size:2rem;">🧠</div>
        <div style="flex:1; min-width:200px;">
            <div style="font-size:0.92rem; font-weight:600; color:#0f172a; margin-bottom:0.3rem;">
                Comment fonctionne RiskRadar ?
            </div>
            <div style="font-size:0.82rem; color:#374151; line-height:1.7;">
                J'ai entraîné une intelligence artificielle sur des millions d'entreprises
                qui ont survécu ou fermé. Elle détecte les signaux de fragilité pour prédire
                ce qui pourrait arriver <strong>dans le futur</strong> — avant que ça arrive.
            </div>
        </div>
        <div style="text-align:center; min-width:90px;">
            <div style="font-size:1.2rem; color:#f59e0b; margin-bottom:2px;">{etoiles(auc_rec)}</div>
            <div style="font-size:1rem; font-weight:700; color:#0f172a;">{nb_correct}/100 réussi</div>
            <div style="font-size:0.65rem; color:#94a3b8;">lors de mes tests</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------
# ZONE DE RECHERCHE
# -------------------------------------------------------
st.markdown(
    """
    <div style="background:#f1f5f9; border:0.5px solid #cbd5e1; border-radius:16px;
                padding:1.4rem 1.8rem;">
        <div style="font-size:1rem; font-weight:600; color:#0f172a; margin-bottom:0.3rem;">
            Analysez une ou plusieurs entreprises
        </div>
        <div style="font-size:0.8rem; color:#64748b; line-height:1.7;">
            Entrez le(s) numéro(s) SIREN — l'identifiant unique à 9 chiffres de chaque entreprise française.<br>
            Plusieurs SIREN séparés par des virgules.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

col_in, col_btn = st.columns([5, 1])
with col_in:
    siren_input = st.text_input(
        "Numéro SIREN",
        placeholder="Ex : 000325175     ou     000325175, 309634954, 552032534",
        label_visibility="collapsed"
    )
with col_btn:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    analyser = st.button("Analyser →")

# -------------------------------------------------------
# TRAITEMENT ET RÉSULTATS
# -------------------------------------------------------
if analyser and siren_input:

    siren_input_nettoye = siren_input.strip()
    if not siren_input_nettoye:
        st.warning("Veuillez entrer au moins un numéro SIREN pour lancer l'analyse.")
        st.stop()

    modele = charger_modele(modeles_data[modele_choisi]["fichier"])
    if modele is None:
        st.error("Modèle introuvable. Veuillez exécuter 07_ml_models.py.")
        st.stop()

    sirens_bruts = [s.strip() for s in siren_input_nettoye.split(",") if s.strip()]
    sirens_valides = [s for s in sirens_bruts if s.isdigit() and len(s) == 9]
    sirens_invalides = [s for s in sirens_bruts if s not in sirens_valides]

    if sirens_invalides:
        st.warning(
            f"Les numéros suivants ont été ignorés — 9 chiffres requis : {', '.join(sirens_invalides)}"
        )
    if not sirens_valides:
        st.error("Aucun numéro SIREN valide. Chaque SIREN doit contenir exactement 9 chiffres.")
        st.stop()

    with st.spinner(f"Analyse de {len(sirens_valides)} entreprise(s) en cours..."):
        resultats = []
        for siren in sirens_valides:
            entreprise = chercher_entreprise_parquet(siren)
            if entreprise is not None:
                X = preparer_features(entreprise)
                proba = modele.predict_proba(X)[0][1]
                resultats.append({"siren": siren, "entreprise": entreprise, "proba": proba})
            else:
                resultats.append({"siren": siren, "entreprise": None, "proba": None})

    st.divider()

    # RÉSUMÉ MULTI SIREN
    if len(resultats) > 1:
        nb_stables = sum(1 for r in resultats if r["proba"] is not None and r["proba"] < 0.35)
        nb_moderes = sum(1 for r in resultats if r["proba"] is not None and 0.35 <= r["proba"] < 0.65)
        nb_risques = sum(1 for r in resultats if r["proba"] is not None and r["proba"] >= 0.65)
        nb_introuvables = sum(1 for r in resultats if r["proba"] is None)
        siren_stables = [r["siren"] for r in resultats if r["proba"] is not None and r["proba"] < 0.35]
        siren_moderes = [r["siren"] for r in resultats if r["proba"] is not None and 0.35 <= r["proba"] < 0.65]
        siren_risques = [r["siren"] for r in resultats if r["proba"] is not None and r["proba"] >= 0.65]

        rs1, rs2, rs3 = st.columns(3)
        rs1.markdown(
            f"""<div style="background:#f0fdf4; border:0.5px solid #86efac;
                border-radius:14px; padding:1.2rem; text-align:center;">
                <div style="font-size:2rem; margin-bottom:0.3rem;">✅</div>
                <div style="font-size:0.95rem; font-weight:700; color:#16a34a;">
                    {nb_stables} entreprise{'s stables' if nb_stables > 1 else ' stable'}
                </div>
                <div style="font-size:0.72rem; color:#64748b; margin-top:4px;">
                    {', '.join(siren_stables) if siren_stables else 'Aucune'}
                </div>
            </div>""",
            unsafe_allow_html=True
        )
        rs2.markdown(
            f"""<div style="background:#fffbeb; border:0.5px solid #fcd34d;
                border-radius:14px; padding:1.2rem; text-align:center;">
                <div style="font-size:2rem; margin-bottom:0.3rem;">⚠️</div>
                <div style="font-size:0.95rem; font-weight:700; color:#d97706;">
                    {nb_moderes} à surveiller
                </div>
                <div style="font-size:0.72rem; color:#64748b; margin-top:4px;">
                    {', '.join(siren_moderes) if siren_moderes else 'Aucune'}
                </div>
            </div>""",
            unsafe_allow_html=True
        )
        rs3.markdown(
            f"""<div style="background:#fef2f2; border:0.5px solid #fca5a5;
                border-radius:14px; padding:1.2rem; text-align:center;">
                <div style="font-size:2rem; margin-bottom:0.3rem;">🚨</div>
                <div style="font-size:0.95rem; font-weight:700; color:#dc2626;">
                    {nb_risques} risque{'s' if nb_risques > 1 else ''} élevé{'s' if nb_risques > 1 else ''}
                </div>
                <div style="font-size:0.72rem; color:#64748b; margin-top:4px;">
                    {', '.join(siren_risques) if siren_risques else 'Aucune'}
                </div>
            </div>""",
            unsafe_allow_html=True
        )

        if nb_introuvables > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(
                f"🔍 {nb_introuvables} numéro{'s' if nb_introuvables > 1 else ''} SIREN "
                f"introuvable{'s' if nb_introuvables > 1 else ''} dans la base INSEE SIRENE."
            )

        st.markdown("<br>", unsafe_allow_html=True)

    # DÉTAIL PAR ENTREPRISE
    for r in resultats:
        siren = r["siren"]
        entreprise = r["entreprise"]
        proba = r["proba"]

        if entreprise is None:
            st.warning(
                f"🔍 SIREN **{siren}** — Introuvable dans la base INSEE SIRENE. "
                f"Vérifiez que le numéro est correct."
            )
            continue

        picto, titre_res, phrase_res, couleur, bg, border, score, label_court = obtenir_verdict(proba)
        anciennete_label, annee_creation = calculer_anciennete(entreprise)
        tranche = str(entreprise.get("trancheEffectifsUniteLegale", "NN"))
        categorie = str(entreprise.get("categorieEntreprise", ""))
        naf_code = str(entreprise.get("activitePrincipaleUniteLegale", "N/A"))
        naf_label = naf_lisible(naf_code)
        try:
            nb_periodes = int(float(entreprise.get("nombrePeriodesUniteLegale", 1)))
        except Exception:
            nb_periodes = 1
        cat_label = LABEL_CATEGORIE.get(categorie, "Non classée") if categorie not in ["nan", ""] else "Non classée"
        taille_label = LABEL_TRANCHES.get(tranche, "Non renseigné")
        certitude = "Très élevée" if abs(score - 50) > 30 else ("Modérée" if abs(score - 50) > 15 else "Faible")
        annee_str = str(annee_creation) if annee_creation else "N/A"
        position_curseur = max(2, min(int(score) - 1, 97))

        # VUE COMPACTE MULTI
        if len(resultats) > 1:
            st.markdown(
                f"""
                <div style="border:0.5px solid {border}; background:{bg};
                            border-radius:14px; padding:1rem; margin-bottom:0.8rem;">
                    <div style="display:grid;
                                grid-template-columns:110px 1fr 1fr 1fr 1fr 180px;
                                gap:12px; align-items:center;">
                        <div style="text-align:center;">
                            <div style="font-size:1.8rem;">{picto}</div>
                            <div style="font-size:0.78rem; font-weight:700; color:{couleur};">
                                {label_court}
                            </div>
                            <div style="font-size:0.62rem; color:#94a3b8;">{siren}</div>
                        </div>
                        <div>
                            <div style="font-size:0.58rem; color:#94a3b8; margin-bottom:2px;">Depuis</div>
                            <div style="font-size:0.88rem; font-weight:600; color:#0f172a;">{anciennete_label}</div>
                        </div>
                        <div>
                            <div style="font-size:0.58rem; color:#94a3b8; margin-bottom:2px;">Salariés</div>
                            <div style="font-size:0.85rem; font-weight:600; color:#0f172a;">{taille_label[:14]}</div>
                        </div>
                        <div>
                            <div style="font-size:0.58rem; color:#94a3b8; margin-bottom:2px;">Secteur</div>
                            <div style="font-size:0.85rem; font-weight:600; color:#0f172a;">{naf_label[:16]}</div>
                        </div>
                        <div>
                            <div style="font-size:0.58rem; color:#94a3b8; margin-bottom:2px;">Évolutions</div>
                            <div style="font-size:0.88rem; font-weight:600; color:#0f172a;">{nb_periodes}</div>
                        </div>
                        <div>
                            <div style="font-size:0.58rem; color:#94a3b8; margin-bottom:4px;">Risque détecté</div>
                            <div style="background:#e2e8f0; border-radius:50px; height:8px;
                                        overflow:hidden; margin-bottom:3px;">
                                <div style="background:{couleur}; width:{int(score)}%;
                                            height:100%; border-radius:50px;"></div>
                            </div>
                            <div style="font-size:0.65rem; color:#64748b;">{label_court} — {score:.1f}%</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # VUE COMPLÈTE 1 SIREN
        else:
            st.markdown(
                f"<div style='font-size:0.62rem; color:#2563eb; text-transform:uppercase; "
                f"letter-spacing:2px; margin-bottom:1rem; font-weight:600;'>"
                f"Analyse du SIREN {siren}</div>",
                unsafe_allow_html=True
            )

            col_g, col_d = st.columns([1, 2])

            with col_g:
                st.markdown(
                    f"""
                    <div style="background:{bg}; border:2px solid {border}; border-radius:18px;
                                padding:2rem 1.5rem; text-align:center; margin-bottom:12px;">
                        <div style="font-size:3rem; margin-bottom:0.8rem;">{picto}</div>
                        <div style="font-size:1.05rem; font-weight:700; color:{couleur};
                                    margin-bottom:0.6rem; line-height:1.3;">{titre_res}</div>
                        <div style="font-size:0.78rem; color:#64748b; line-height:1.6;
                                    margin-bottom:1rem;">{phrase_res}</div>
                        <div style="background:#e2e8f0; border-radius:50px; height:8px;
                                    overflow:hidden; margin-bottom:5px;">
                            <div style="background:{couleur}; width:{int(score)}%;
                                        height:100%; border-radius:50px;"></div>
                        </div>
                        <div style="font-size:0.65rem; color:#94a3b8;">
                            Risque détecté : {score:.1f} / 100
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div style="background:white; border:0.5px solid #e2e8f0; border-radius:12px;
                                padding:1rem; text-align:center; margin-bottom:12px;">
                        <div style="font-size:0.6rem; color:#94a3b8; text-transform:uppercase;
                                    letter-spacing:1px; margin-bottom:5px;">
                            Fiabilité de cette prédiction
                        </div>
                        <div style="font-size:1.1rem; color:#f59e0b; margin-bottom:4px;">
                            {etoiles(auc_rec)}
                        </div>
                        <div style="font-size:1.5rem; font-weight:700; color:#2563eb;">
                            {nb_correct}
                            <span style="font-size:0.85rem; color:#94a3b8; font-weight:400;">
                                bonnes prédictions sur 100
                            </span>
                        </div>
                        <div style="font-size:0.7rem; color:#94a3b8; margin-top:3px;">
                            lors de mes tests sur des entreprises réelles
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div style="background:white; border:0.5px solid #e2e8f0;
                                border-radius:10px; padding:0.8rem;">
                        <div style="font-size:0.6rem; color:#2563eb; text-transform:uppercase;
                                    letter-spacing:1px; font-weight:600; margin-bottom:8px;">
                            Position parmi toutes les entreprises
                        </div>
                        <div style="position:relative; height:16px;
                                    background:linear-gradient(90deg,#16a34a 0%,#d97706 50%,#dc2626 100%);
                                    border-radius:50px; margin-bottom:0.4rem;">
                            <div style="position:absolute; left:{position_curseur}%; top:-5px;
                                        width:3px; height:26px; background:#0f172a;
                                        border-radius:2px; opacity:0.9;"></div>
                        </div>
                        <div style="display:flex; justify-content:space-between;
                                    font-size:0.58rem; color:#94a3b8;">
                            <span>Très peu risquée</span>
                            <span>À surveiller</span>
                            <span>Très risquée</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_d:
                st.markdown(
                    "<div style='font-size:0.62rem; color:#2563eb; text-transform:uppercase; "
                    "letter-spacing:2px; margin-bottom:1rem; font-weight:600;'>"
                    "Fiche entreprise</div>",
                    unsafe_allow_html=True
                )

                d1, d2, d3 = st.columns(3)
                d1.metric("En activité depuis", anciennete_label)
                d1.metric("Nombre de salariés", taille_label)
                d2.metric("Taille", cat_label)
                d2.metric("Secteur d'activité", naf_label)
                d3.metric("Évolutions enregistrées", str(nb_periodes))
                d3.metric("Certitude de l'analyse", certitude)

                st.markdown(
                    f"""
                    <div style="background:#eff6ff; border:0.5px solid #bfdbfe;
                                border-radius:10px; padding:1rem; margin:10px 0;">
                        <div style="font-size:0.82rem; color:#1e40af; line-height:1.7;">
                            <strong>Ce que cette prédiction signifie concrètement :</strong>
                            J'ai comparé cette entreprise avec des millions d'entreprises qui ont
                            survécu ou fermé grâce à mon modèle d'intelligence artificielle.
                            Son profil global — ancienneté, secteur d'activité, taille, nombre
                            d'évolutions enregistrées et autres caractéristiques administratives —
                            correspond à celui des entreprises classées
                            <strong>{label_court.lower()}</strong> dans mes données historiques.
                        </div>
                    </div>                    
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div style="background:white; border:0.5px solid #e2e8f0;
                                border-radius:10px; padding:1rem;">
                        <div style="font-size:0.6rem; color:#2563eb; text-transform:uppercase;
                                    letter-spacing:1.5px; margin-bottom:0.8rem; font-weight:600;">
                            Évolutions enregistrées depuis la création
                        </div>
                        <div style="font-size:0.82rem; color:#0f172a; line-height:1.7;">
                            Cette entreprise a connu <strong>{nb_periodes} évolution(s)</strong>
                            depuis sa création en {annee_str}. Chaque évolution peut correspondre
                            à un changement de statut, de secteur d'activité, de forme juridique,
                            de taille, de dirigeant ou de dénomination sociale.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

elif analyser and not siren_input:
    st.warning("Veuillez entrer au moins un numéro SIREN pour lancer l'analyse.")

# -------------------------------------------------------
# PIED DE PAGE
# -------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.caption(
    f"RiskRadar — Mémoire de recherche — Mastère 2 Big Data, IA et Dév — École IPSSI 2026 | "
    f"Données INSEE SIRENE / data.gouv.fr — {formater_kpi(nb_entreprises)} entreprises analysées"
)