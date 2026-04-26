"""
Script de preparation des donnees SIRENE - ETAPE SILVER
Objectif : extraire et nettoyer les colonnes utiles pour le ML
Source : data/raw/sirene_entreprises_bronze_latest.parquet
Sortie : data/silver/sirene_entreprises_silver_YYYY-MM-DD.parquet
"""

import os
import shutil
import pandas as pd
from datetime import datetime

DATE_EXECUTION = datetime.now().strftime("%Y-%m-%d")

# Chemins des fichiers source et de sortie
DOSSIER_SILVER = "data/silver"
CHEMIN_BRONZE = "data/raw/sirene_entreprises_bronze_latest.parquet"
CHEMIN_FINAL = os.path.join(DOSSIER_SILVER, f"sirene_entreprises_silver_{DATE_EXECUTION}.parquet")
CHEMIN_LATEST = os.path.join(DOSSIER_SILVER, "sirene_entreprises_silver_latest.parquet")
CHEMIN_LOG = os.path.join(DOSSIER_SILVER, f"silver_log_{DATE_EXECUTION}.txt")

os.makedirs(DOSSIER_SILVER, exist_ok=True)

# Colonnes retenues pour la prediction de defaillance
# Selectionnees selon leur taux de completude et leur valeur predictive
COLONNES_SILVER = [
    "siren",
    "etatAdministratifUniteLegale",
    "dateCreationUniteLegale",
    "categorieJuridiqueUniteLegale",
    "activitePrincipaleUniteLegale",
    "trancheEffectifsUniteLegale",
    "categorieEntreprise",
    "economieSocialeSolidaireUniteLegale",
    "nombrePeriodesUniteLegale",
    "dateDebut"
]


def log(message):
    # Ecriture horodatee dans le terminal et dans le fichier log
    horodatage = datetime.now().strftime("%H:%M:%S")
    ligne = f"[{horodatage}] {message}"
    print(ligne)
    with open(CHEMIN_LOG, "a", encoding="utf-8") as f:
        f.write(ligne + "\n")


def verifier_source():
    # Verification de l'existence du fichier bronze avant tout traitement
    if not os.path.exists(CHEMIN_BRONZE):
        log(f"ERREUR : Fichier bronze non trouve : {CHEMIN_BRONZE}")
        return False
    taille_mb = os.path.getsize(CHEMIN_BRONZE) / (1024 * 1024)
    log(f"Fichier bronze trouve : {taille_mb:.2f} MB")
    return True


def transformer_silver():
    log("Chargement du fichier bronze...")

    # Lecture uniquement des colonnes utiles pour limiter la consommation memoire
    df = pd.read_parquet(CHEMIN_BRONZE, columns=COLONNES_SILVER)
    log(f"Lignes chargees : {len(df):,}")

    # Suppression des lignes sans identifiant SIREN ou sans etat administratif
    # Ces lignes sont inexploitables pour l'entrainement des modeles ML
    nb_avant = len(df)
    df = df[df["siren"].notna() & df["etatAdministratifUniteLegale"].notna()]
    nb_apres = len(df)
    log(f"Lignes supprimees (nulls cles) : {nb_avant - nb_apres:,}")
    log(f"Lignes conservees : {nb_apres:,}")

    # Distribution de la variable cible avant sauvegarde
    # A = Active, C = Cessee
    distribution = df["etatAdministratifUniteLegale"].value_counts(dropna=False)
    log("Distribution etatAdministratifUniteLegale :")
    for valeur, nb in distribution.items():
        log(f"  {valeur} : {nb:,}")

    # Sauvegarde du fichier Silver au format Parquet
    log("Sauvegarde du fichier Silver...")
    df.to_parquet(CHEMIN_FINAL, engine="pyarrow", index=False)

    taille_mb = os.path.getsize(CHEMIN_FINAL) / (1024 * 1024)
    log(f"Fichier Silver sauvegarde : {taille_mb:.2f} MB")
    log(f"Colonnes conservees : {len(df.columns)}")

    return True


print("\n" + "="*60)
print("PREPARATION DES DONNEES SIRENE - SILVER")
print(f"Date : {DATE_EXECUTION}")
print("="*60)

log("Demarrage du script Silver")

if not verifier_source():
    exit()

succes = transformer_silver()
if not succes:
    exit()

# Copie du fichier date vers le fichier latest pour les etapes suivantes
shutil.copy2(CHEMIN_FINAL, CHEMIN_LATEST)
log(f"Fichier latest mis a jour : {CHEMIN_LATEST}")

log("PREPARATION SILVER COMPLETE")

# Suppression du log apres execution reussie
# Le log est conserve uniquement en cas d'erreur pour faciliter le diagnostic
if os.path.exists(CHEMIN_LOG):
    os.remove(CHEMIN_LOG)

print("\n" + "="*60)
print(f"Date fin : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("="*60)