"""
Script d'entrainement des modeles ML - PREDICTION DE DEFAILLANCE
Objectif : entrainer et evaluer trois modeles de classification
Source : data/gold/sirene_entreprises_gold_latest.parquet
Modeles : Logistic Regression (baseline), Random Forest, XGBoost (principal)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)
from xgboost import XGBClassifier
import joblib

DATE_EXECUTION = datetime.now().strftime("%Y-%m-%d")

# Chemins des fichiers source et de sortie
DOSSIER_MODELES = "data/modeles"
CHEMIN_GOLD = "data/gold/sirene_entreprises_gold_latest.parquet"
CHEMIN_LOG = os.path.join(DOSSIER_MODELES, f"ml_log_{DATE_EXECUTION}.txt")

# Proportion du jeu de test : 20% pour le test, 80% pour l'entrainement
TAILLE_TEST = 0.2

# Graine aleatoire fixe pour la reproductibilite des resultats
GRAINE = 42

# Proportion du jeu de donnees utilisee pour l'entrainement
# 10% suffit pour 29M de lignes et reduit considerablement le temps de calcul
PROPORTION_ECHANTILLON = 0.1

os.makedirs(DOSSIER_MODELES, exist_ok=True)


def log(message):
    # Ecriture horodatee dans le terminal et dans le fichier log
    horodatage = datetime.now().strftime("%H:%M:%S")
    ligne = f"[{horodatage}] {message}"
    print(ligne)
    with open(CHEMIN_LOG, "a", encoding="utf-8") as f:
        f.write(ligne + "\n")


def afficher_resultats(nom_modele, y_test, y_pred, y_pred_proba):
    # Affichage des metriques d'evaluation du modele
    # L'AUC-ROC est la metrique principale car elle mesure la capacite
    # du modele a distinguer les entreprises actives des cessees
    log(f"\n{'='*50}")
    log(f"RESULTATS : {nom_modele}")
    log(f"{'='*50}")
    log(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    log(f"AUC-ROC  : {roc_auc_score(y_test, y_pred_proba):.4f}")
    log("Matrice de confusion :")
    matrice = confusion_matrix(y_test, y_pred)
    log(f"  Vrais Negatifs (TN) : {matrice[0][0]:,}")
    log(f"  Faux Positifs  (FP) : {matrice[0][1]:,}")
    log(f"  Faux Negatifs  (FN) : {matrice[1][0]:,}")
    log(f"  Vrais Positifs (TP) : {matrice[1][1]:,}")
    log("Rapport de classification :")
    rapport = classification_report(y_test, y_pred, target_names=["Active (0)", "Cessee (1)"])
    for ligne_rapport in rapport.split("\n"):
        log(f"  {ligne_rapport}")


print("\n" + "="*60)
print("ENTRAINEMENT DES MODELES ML - PREDICTION DEFAILLANCE")
print(f"Date : {DATE_EXECUTION}")
print("="*60)

log("Demarrage du script ML")

# Verification de l'existence du fichier Gold
if not os.path.exists(CHEMIN_GOLD):
    log(f"ERREUR : Fichier Gold non trouve : {CHEMIN_GOLD}")
    exit()

# Chargement du fichier Gold
log("Chargement du fichier Gold...")
df = pd.read_parquet(CHEMIN_GOLD)
log(f"Lignes chargees : {len(df):,}")

# Echantillonnage stratifie pour reduire le temps de calcul
# L'echantillonnage stratifie preserve la proportion A/C dans l'echantillon
log(f"Echantillonnage stratifie : {PROPORTION_ECHANTILLON*100:.0f}% des donnees...")
df_actives = df[df["cible"] == 0].sample(frac=PROPORTION_ECHANTILLON, random_state=GRAINE)
df_cessees = df[df["cible"] == 1].sample(frac=PROPORTION_ECHANTILLON, random_state=GRAINE)
df = pd.concat([df_actives, df_cessees]).reset_index(drop=True)
log(f"Lignes apres echantillonnage : {len(df):,}")
log(f"Distribution cible - Actives (0) : {(df['cible'] == 0).sum():,} | Cessees (1) : {(df['cible'] == 1).sum():,}")

# Separation des features et de la variable cible
X = df.drop(columns=["cible"])
y = df["cible"]

log(f"Features utilisees : {list(X.columns)}")

# Division en jeux d'entrainement et de test
# La stratification garantit la meme proportion A/C dans les deux jeux
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TAILLE_TEST,
    random_state=GRAINE,
    stratify=y
)
log(f"Taille jeu entrainement : {len(X_train):,}")
log(f"Taille jeu test : {len(X_test):,}")

# -----------------------------------------------------------
# MODELE 1 : LOGISTIC REGRESSION (baseline)
# Modele simple et interpretable utilise comme reference
# -----------------------------------------------------------
log("\nEntrainement Logistic Regression (baseline)...")
modele_lr = LogisticRegression(max_iter=1000, random_state=GRAINE)
modele_lr.fit(X_train, y_train)
y_pred_lr = modele_lr.predict(X_test)
y_proba_lr = modele_lr.predict_proba(X_test)[:, 1]
afficher_resultats("LOGISTIC REGRESSION", y_test, y_pred_lr, y_proba_lr)
joblib.dump(modele_lr, os.path.join(DOSSIER_MODELES, "modele_logistic_regression.pkl"))
log("Modele Logistic Regression sauvegarde")

# -----------------------------------------------------------
# MODELE 2 : RANDOM FOREST
# Modele intermediaire base sur un ensemble d'arbres de decision
# -----------------------------------------------------------
log("\nEntrainement Random Forest...")
modele_rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=8,
    max_samples=0.5,
    random_state=GRAINE,
    n_jobs=-1
)
modele_rf.fit(X_train, y_train)
y_pred_rf = modele_rf.predict(X_test)
y_proba_rf = modele_rf.predict_proba(X_test)[:, 1]
afficher_resultats("RANDOM FOREST", y_test, y_pred_rf, y_proba_rf)
joblib.dump(modele_rf, os.path.join(DOSSIER_MODELES, "modele_random_forest.pkl"))
log("Modele Random Forest sauvegarde")

# -----------------------------------------------------------
# MODELE 3 : XGBOOST (modele principal)
# Modele de gradient boosting, generalement le plus performant
# sur des donnees tabulaires de grande taille
# -----------------------------------------------------------
log("\nEntrainement XGBoost (modele principal)...")
modele_xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=GRAINE,
    n_jobs=-1,
    eval_metric="logloss",
    verbosity=0
)
modele_xgb.fit(X_train, y_train)
y_pred_xgb = modele_xgb.predict(X_test)
y_proba_xgb = modele_xgb.predict_proba(X_test)[:, 1]
afficher_resultats("XGBOOST", y_test, y_pred_xgb, y_proba_xgb)
joblib.dump(modele_xgb, os.path.join(DOSSIER_MODELES, "modele_xgboost.pkl"))
log("Modele XGBoost sauvegarde")

# -----------------------------------------------------------
# COMPARAISON FINALE ET SAUVEGARDE DES PERFORMANCES
# -----------------------------------------------------------
log(f"\n{'='*50}")
log("COMPARAISON FINALE DES MODELES")
log(f"{'='*50}")
log(f"{'Modele':<25} {'Accuracy':>10} {'AUC-ROC':>10}")
log(f"{'-'*45}")
log(f"{'Logistic Regression':<25} {accuracy_score(y_test, y_pred_lr):>10.4f} {roc_auc_score(y_test, y_proba_lr):>10.4f}")
log(f"{'Random Forest':<25} {accuracy_score(y_test, y_pred_rf):>10.4f} {roc_auc_score(y_test, y_proba_rf):>10.4f}")
log(f"{'XGBoost':<25} {accuracy_score(y_test, y_pred_xgb):>10.4f} {roc_auc_score(y_test, y_proba_xgb):>10.4f}")

# Sauvegarde des performances dans un fichier JSON
# Ce fichier est lu par le dashboard pour afficher les metriques a jour
# et determiner automatiquement le modele recommande
import json

performances = {
    "date_entrainement": DATE_EXECUTION,
    "modeles": {
        "XGBoost": {
            "fichier": "data/modeles/modele_xgboost.pkl",
            "accuracy": round(accuracy_score(y_test, y_pred_xgb) * 100, 2),
            "auc_roc": round(roc_auc_score(y_test, y_proba_xgb) * 100, 2),
            "description": "Modele principal - Gradient Boosting"
        },
        "Random Forest": {
            "fichier": "data/modeles/modele_random_forest.pkl",
            "accuracy": round(accuracy_score(y_test, y_pred_rf) * 100, 2),
            "auc_roc": round(roc_auc_score(y_test, y_proba_rf) * 100, 2),
            "description": "Ensemble d'arbres de decision"
        },
        "Logistic Regression": {
            "fichier": "data/modeles/modele_logistic_regression.pkl",
            "accuracy": round(accuracy_score(y_test, y_pred_lr) * 100, 2),
            "auc_roc": round(roc_auc_score(y_test, y_proba_lr) * 100, 2),
            "description": "Modele de reference lineaire"
        }
    }
}

# Determination automatique du modele recommande selon l'AUC-ROC
meilleur_modele = max(
    performances["modeles"],
    key=lambda x: performances["modeles"][x]["auc_roc"]
)
for nom in performances["modeles"]:
    performances["modeles"][nom]["recommande"] = (nom == meilleur_modele)

chemin_performances = os.path.join(DOSSIER_MODELES, "performances.json")
with open(chemin_performances, "w", encoding="utf-8") as f:
    json.dump(performances, f, ensure_ascii=False, indent=2)

log(f"Performances sauvegardees : {chemin_performances}")
log(f"Modele recommande : {meilleur_modele}")
log("\nENTRAINEMENT ML COMPLETE")

print("\n" + "="*60)
print(f"Date fin : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("="*60)