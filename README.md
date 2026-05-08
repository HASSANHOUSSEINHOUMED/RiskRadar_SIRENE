# 📡 RiskRadar — Prédiction de défaillance d'entreprises

> Mémoire de recherche — Mastère 2 Big Data, IA et Dév — École IPSSI 2026

![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Modèle_Principal-FF6600?style=flat-square)
![INSEE](https://img.shields.io/badge/Source-INSEE_SIRENE-0055A4?style=flat-square)
![Parquet](https://img.shields.io/badge/Stockage-Parquet-50ABF1?style=flat-square)
![Statut](https://img.shields.io/badge/Statut-Soutenance_2026-green?style=flat-square)

---

## 🚀 Application en ligne

[![RiskRadar](https://img.shields.io/badge/🌐_RiskRadar-Accéder_à_l'app-2563eb?style=for-the-badge)](https://riskradar-sirene.streamlit.app)

**Données** : [![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Dataset_SIRENE-FFD21E?style=for-the-badge)](https://huggingface.co/datasets/HassanHH2910/riskradar-sirene)

---

## 🎯 Présentation du projet

**RiskRadar** est un système complet de prédiction de défaillance d'entreprises françaises, développé dans le cadre de mon mémoire de recherche en Mastère 2 Big Data, IA et Dév.

J'ai conçu un pipeline de données de bout en bout — de la collecte des données officielles INSEE jusqu'au dashboard interactif — capable d'analyser **29,5 millions d'entreprises françaises** et de prédire leur risque futur de fermeture grâce au Machine Learning.

### 💡 Ce que fait RiskRadar

- **Analyse n'importe quelle entreprise française** à partir de son numéro SIREN
- **Prédit le risque futur de fermeture** avec 95% de précision
- **Compare simultanément plusieurs entreprises** en quelques secondes
- **Se met à jour automatiquement** chaque mois avec les nouvelles données INSEE

---

## 📊 Résultats des modèles

| Modèle | Accuracy | AUC-ROC | Statut |
|--------|----------|---------|--------|
| Logistic Regression | 78% | 84% | Baseline |
| Random Forest | 93% | 98% | Intermédiaire |
| **XGBoost** | **95%** | **99%** | ✅ **Recommandé** |

> Entraînement sur **2,93 millions d'entreprises** (10% stratifié des 29,5M) — contrainte RAM documentée et défendable.

---

## 🏗️ Architecture du pipeline

```
data.gouv.fr / INSEE SIRENE
        ↓
┌───────────────────────────────────────────────────┐
│  BRONZE  │  Téléchargement mensuel automatique    │
│          │  29,5M entreprises — 655 MB Parquet    │
└───────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────┐
│  SILVER  │  10 colonnes utiles extraites          │
│          │  29,5M lignes — 327 MB Parquet         │
└───────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────┐
│  GOLD    │  Encodage + Normalisation + Cible      │
│          │  29,5M lignes — 184 MB Parquet         │
└───────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────┐
│   ML     │  XGBoost / Random Forest / LR          │
│          │  Sélection automatique du meilleur     │
└───────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────┐
│  DASHBOARD  │  RiskRadar — Streamlit              │
│             │  https://riskradar-sirene.streamlit.app │
└───────────────────────────────────────────────────┘
```

---

## 🛠️ Stack technique

### Données & Pipeline
![PyArrow](https://img.shields.io/badge/PyArrow-Lecture_RAM_optimisée-blue?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-Transformations-150458?style=flat-square&logo=pandas&logoColor=white)
![Parquet](https://img.shields.io/badge/Parquet-Stockage_exclusif-50ABF1?style=flat-square)

### Machine Learning
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Modèles-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Principal-FF6600?style=flat-square)
![RandomForest](https://img.shields.io/badge/Random_Forest-Intermédiaire-228B22?style=flat-square)

### Dashboard & Déploiement
![Streamlit](https://img.shields.io/badge/Streamlit-Interface-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Graphiques-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Hugging_Face-Données-FFD21E?style=flat-square)

---

## 📂 Structure du projet

```
RiskRadar_SIRENE/
│
├── 01_data_preparation_bronze.py   # Téléchargement INSEE avec reprise automatique
├── 02_data_quality_bronze.py       # Validation qualité Bronze
├── 03_data_preparation_silver.py   # Extraction des colonnes utiles
├── 04_data_quality_silver.py       # Validation qualité Silver
├── 05_data_preparation_gold.py     # Encodage, normalisation, variable cible
├── 06_data_quality_gold.py         # Validation qualité Gold
├── 07_ml_models.py                 # Entraînement XGBoost, RF, LR
├── 08_data_quality_modele.py       # Validation et comparaison des modèles
│
├── app.py                          # Dashboard RiskRadar (Streamlit)
│
├── requirements.txt                # Dépendances Python
├── .gitignore                      # Données et modèles exclus
└── README.md                       # Documentation
```

---

## 🚀 Installation et lancement en local

### 1. Cloner le répertoire
```bash
git clone https://github.com/HASSANHOUSSEINHOUMED/RiskRadar_SIRENE.git
cd RiskRadar_SIRENE
```

### 2. Créer l'environnement virtuel
```bash
python -m venv prediction_defaillance_env
prediction_defaillance_env\Scripts\activate
pip install -r requirements.txt
```

### 3. Lancer le pipeline complet
```bash
python 01_data_preparation_bronze.py
python 02_data_quality_bronze.py
python 03_data_preparation_silver.py
python 04_data_quality_silver.py
python 05_data_preparation_gold.py
python 06_data_quality_gold.py
python 07_ml_models.py
python 08_data_quality_modele.py
```

### 4. Lancer le dashboard
```bash
python -m streamlit run app.py
```

> 💡 En production, l'app télécharge automatiquement les données depuis Hugging Face au démarrage.

---

## 🔍 Fonctionnalités du dashboard

### Analyse d'une entreprise
Entrez un numéro SIREN à 9 chiffres pour obtenir :
- ✅ Verdict clair (Stable / À surveiller / Risque élevé)
- 📊 Score de risque sur 100
- 🏢 Fiche complète (ancienneté, secteur, taille, évolutions)
- 📈 Position parmi toutes les entreprises françaises

### Analyse multi-entreprises
Entrez plusieurs SIREN séparés par des virgules pour :
- 🟢 Résumé coloré par catégorie de risque
- 📋 Vue compacte comparative

---

## ⚙️ Décisions techniques clés

| Décision | Justification |
|----------|---------------|
| **Fichier stock Parquet** au lieu de l'API INSEE | API instable et soumise au rate limiting — même données, source officielle |
| **PyArrow** pour la lecture | Évite l'erreur `ArrowMemoryError` sur 8 Go de RAM |
| **10% stratifié** pour l'entraînement | Contrainte RAM documentée — représentativité garantie par la stratification |
| **Parquet exclusivement** | CSV banni — performance et compression supérieures |
| **Sélection automatique du meilleur modèle** | `performances.json` mis à jour à chaque entraînement |
| **Hugging Face** pour le stockage des données | Fichier Silver 327 MB dépasse la limite GitHub de 100 MB |
| **Streamlit Cloud** pour le déploiement | Gratuit, lié directement au GitHub, redéploiement automatique |

---

## 📋 Source des données

Les données proviennent de la base officielle **INSEE SIRENE** publiée sur [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/base-sirene-des-entreprises-et-de-leurs-etablissements-siren-siret/).

- **29,5 millions** d'entreprises françaises
- Actives et fermées **depuis 1973**
- Mise à jour **mensuelle possible**
- Licence **Ouverte / Open Licence**
- Données disponibles sur Hugging Face : [HassanHH2910/riskradar-sirene](https://huggingface.co/datasets/HassanHH2910/riskradar-sirene)

---

## 👤 Auteur

**Hassan HOUSSEIN HOUMED**
Mastère 2 Big Data, IA et Dév — École IPSSI 2026
Encadrant : M. Sayf Bejaoui

📧 hassan.houssein.houmed@gmail.com
🐙 [GitHub](https://github.com/HASSANHOUSSEINHOUMED)
🌐 [RiskRadar](https://riskradar-sirene.streamlit.app)

---

<div align="center">

**Mémoire de recherche — École IPSSI 2026**
*Données officielles INSEE SIRENE / data.gouv.fr*

</div>