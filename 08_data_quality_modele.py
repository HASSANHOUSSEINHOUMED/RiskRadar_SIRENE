"""
Validation de la qualite des modeles ML entraines
Verifie les performances et selectionne le meilleur modele
Source : data/modeles/performances.json
"""

import json
import os
import joblib
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    confusion_matrix, classification_report
)

CHEMIN_PERFORMANCES = "data/modeles/performances.json"
CHEMIN_GOLD = "data/gold/sirene_entreprises_gold_latest.parquet"
CHEMIN_MODELES = "data/modeles"

NOMS_FEATURES = {
    "categorieJuridiqueUniteLegale": "Forme juridique",
    "activitePrincipaleUniteLegale": "Secteur d activite (NAF)",
    "trancheEffectifsUniteLegale": "Taille effectifs",
    "categorieEntreprise": "Categorie entreprise",
    "economieSocialeSolidaireUniteLegale": "Economie sociale et solidaire",
    "nombrePeriodesUniteLegale": "Nombre d evolutions",
    "anciennete_annees": "Anciennete",
    "duree_periode_actuelle_annees": "Duree periode actuelle"
}

print("=" * 60)
print("ANALYSE DE LA QUALITE - MODELES ML")
print(f"Date : {datetime.now().strftime('%Y-%m-%d')}")
print("=" * 60)

# -------------------------------------------------------
# CHARGEMENT PERFORMANCES
# -------------------------------------------------------
if not os.path.exists(CHEMIN_PERFORMANCES):
    print("[ERREUR] Fichier performances.json introuvable.")
    print("Executez d abord 07_ml_models.py.")
    exit(1)

with open(CHEMIN_PERFORMANCES, "r", encoding="utf-8") as f:
    performances = json.load(f)

modeles_data = performances["modeles"]
date_entrainement = performances.get("date_entrainement", "N/A")
echantillon = performances.get("nb_lignes_entrainement", "N/A")

print(f"\n[INFO] Date d entrainement : {date_entrainement}")
print(f"[INFO] Lignes utilisees : {echantillon}")

# -------------------------------------------------------
# PERFORMANCES PAR MODELE
# -------------------------------------------------------
print("\n" + "-" * 60)
print("PERFORMANCES PAR MODELE :")
print("-" * 60)

modele_recommande = None
meilleur_auc = 0

for nom, data in modeles_data.items():
    auc = data.get("auc_roc", 0)
    acc = data.get("accuracy", 0)
    recommande = data.get("recommande", False)
    fichier = data.get("fichier", "")

    if recommande:
        modele_recommande = nom

    if auc > meilleur_auc:
        meilleur_auc = auc

    statut = "RECOMMANDE" if recommande else ""
    print(f"\n  {nom} {statut}")
    print(f"    Accuracy  : {acc:.4f} ({int(acc)} bonnes predictions sur 100)")
    print(f"    AUC-ROC   : {auc:.4f}")

    if os.path.exists(fichier):
        taille = os.path.getsize(fichier) / (1024 * 1024)
        print(f"    Fichier   : {fichier} ({taille:.2f} MB) — OK")
    else:
        print(f"    Fichier   : {fichier} — INTROUVABLE")

# -------------------------------------------------------
# VERIFICATION MODELE RECOMMANDE
# -------------------------------------------------------
print("\n" + "-" * 60)
print("VERIFICATION DU MODELE RECOMMANDE :")
print("-" * 60)

if modele_recommande:
    print(f"  Modele selectionne : {modele_recommande}")
    print(f"  AUC-ROC            : {modeles_data[modele_recommande]['auc_roc']:.4f}")
    print(f"  Accuracy           : {modeles_data[modele_recommande]['accuracy']:.4f}")

    auc_recommande = modeles_data[modele_recommande]["auc_roc"]
    if auc_recommande == meilleur_auc:
        print("  RESULTAT : Le modele recommande est bien le plus performant")
    else:
        print("  ATTENTION : Le modele recommande n est pas le plus performant")

# -------------------------------------------------------
# IMPORTANCE DES VARIABLES - XGBOOST
# -------------------------------------------------------
print("\n" + "-" * 60)
print("IMPORTANCE DES VARIABLES (XGBoost) :")
print("-" * 60)

if modele_recommande:
    fichier_modele = modeles_data[modele_recommande]["fichier"]
    if os.path.exists(fichier_modele):
        modele = joblib.load(fichier_modele)

        # XGBoost expose feature_importances_ via gain
        if hasattr(modele, "feature_importances_"):
            features = list(NOMS_FEATURES.keys())
            importances = modele.feature_importances_

            # Tri par importance decroissante
            indices = np.argsort(importances)[::-1]

            print(f"\n  {'Rang':<6} {'Variable':<40} {'Importance':>12}")
            print(f"  {'-'*6} {'-'*40} {'-'*12}")
            for rang, idx in enumerate(indices):
                nom_court = features[idx] if idx < len(features) else f"feature_{idx}"
                nom_lisible = NOMS_FEATURES.get(nom_court, nom_court)
                pct = importances[idx] * 100
                print(f"  {rang+1:<6} {nom_lisible:<40} {pct:>11.2f}%")

            # Variable la plus importante
            plus_importante = NOMS_FEATURES.get(features[indices[0]], features[indices[0]])
            print(f"\n  Variable la plus determinante : {plus_importante} ({importances[indices[0]]*100:.2f}%)")
            print(f"  Interpretation : le modele s appuie principalement sur cette")
            print(f"  caracteristique pour evaluer le risque de fermeture definitive.")
        else:
            print("  Feature importance non disponible pour ce modele")

# -------------------------------------------------------
# VALIDATION SUR ECHANTILLON GOLD
# -------------------------------------------------------
print("\n" + "-" * 60)
print("VALIDATION SUR ECHANTILLON GOLD (1% stratifie) :")
print("-" * 60)

if modele_recommande and os.path.exists(CHEMIN_GOLD):
    print("  Chargement d un echantillon de validation...")

    colonnes = [
        "categorieJuridiqueUniteLegale", "activitePrincipaleUniteLegale",
        "trancheEffectifsUniteLegale", "categorieEntreprise",
        "economieSocialeSolidaireUniteLegale", "nombrePeriodesUniteLegale",
        "anciennete_annees", "duree_periode_actuelle_annees", "cible"
    ]
    df = pq.read_table(CHEMIN_GOLD, columns=colonnes).to_pandas()

    df_val = df.groupby("cible", group_keys=False).apply(
        lambda x: x.sample(frac=0.01, random_state=42)
    ).reset_index(drop=True)

    y_val = df_val["cible"].reset_index(drop=True)
    X_val = df_val.drop(columns=["cible"]).reset_index(drop=True)

    fichier_modele = modeles_data[modele_recommande]["fichier"]
    modele = joblib.load(fichier_modele)

    y_pred = modele.predict(X_val)
    y_proba = modele.predict_proba(X_val)[:, 1]

    acc_val = accuracy_score(y_val, y_pred)
    auc_val = roc_auc_score(y_val, y_proba)
    cm = confusion_matrix(y_val, y_pred)

    print(f"  Lignes de validation : {len(df_val):,}")
    print(f"  Accuracy             : {acc_val:.4f}")
    print(f"  AUC-ROC              : {auc_val:.4f}")
    print(f"\n  Matrice de confusion :")
    print(f"    Vrais Negatifs (TN) : {cm[0][0]:,}")
    print(f"    Faux Positifs  (FP) : {cm[0][1]:,}")
    print(f"    Faux Negatifs  (FN) : {cm[1][0]:,}")
    print(f"    Vrais Positifs (TP) : {cm[1][1]:,}")
    print(f"\n  Rapport de classification :")
    print(classification_report(y_val, y_pred,
          target_names=["Active (0)", "Cessee (1)"]))

    auc_enregistre = modeles_data[modele_recommande]["auc_roc"] / 100
    auc_val_pct = auc_val * 100
    ecart = abs(auc_val - auc_enregistre)
    print(f"  AUC-ROC enregistre   : {auc_enregistre*100:.2f}%")
    print(f"  AUC-ROC validation   : {auc_val_pct:.2f}%")
    print(f"  Ecart                : {ecart:.4f}")

    if ecart < 0.05:
        print("  RESULTAT : Modele stable — performances coherentes")
    else:
        print("  ATTENTION : Ecart important — verifier le pipeline")

# -------------------------------------------------------
# SYNTHESE
# -------------------------------------------------------
print("\n" + "-" * 60)
print("SYNTHESE :")
print("-" * 60)
for nom, data in modeles_data.items():
    recommande = "← RECOMMANDE" if data.get("recommande") else ""
    print(f"  {nom} | AUC-ROC : {data['auc_roc']:.4f} | Accuracy : {data['accuracy']:.4f} {recommande}")

print("\n" + "=" * 60)
print("Analyse complete")
print(f"Date fin : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("=" * 60)