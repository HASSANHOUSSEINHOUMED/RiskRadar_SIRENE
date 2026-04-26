"""
Script de verification de la qualite des donnees SIRENE - ETAPE SILVER
Objectif : valider l'integrite du fichier Silver avant transformation Gold
Source : data/silver/sirene_entreprises_silver_latest.parquet
"""

import os
import pandas as pd
from datetime import datetime

DATE_EXECUTION = datetime.now().strftime("%Y-%m-%d")

# Chemin vers le fichier Silver produit lors de l'etape precedente
CHEMIN_SILVER = "data/silver/sirene_entreprises_silver_latest.parquet"
CHEMIN_LOG = os.path.join("data/silver", f"silver_quality_log_{DATE_EXECUTION}.txt")


def log(message):
    # Ecriture horodatee dans le terminal et dans le fichier log
    horodatage = datetime.now().strftime("%H:%M:%S")
    ligne = f"[{horodatage}] {message}"
    print(ligne)
    with open(CHEMIN_LOG, "a", encoding="utf-8") as f:
        f.write(ligne + "\n")


# Verification de l'existence du fichier Silver avant tout traitement
if not os.path.exists(CHEMIN_SILVER):
    print(f"ERREUR : Fichier {CHEMIN_SILVER} non trouve")
    exit()

print("Chargement des donnees Silver...")
df = pd.read_parquet(CHEMIN_SILVER)

print("\n" + "="*60)
print("ANALYSE DE LA QUALITE - DONNEES SILVER")
print("="*60)

# Dimensions generales du jeu de donnees Silver
log(f"Nombre total de lignes : {len(df):,}")
log(f"Nombre total de colonnes : {len(df.columns)}")
log(f"Colonnes : {list(df.columns)}")

print("\n" + "-"*60)
print("COMPLETUDE PAR COLONNE :")
print("-"*60)

# Calcul du taux de remplissage pour chaque colonne retenue
for colonne in df.columns:
    total = len(df)
    non_null = df[colonne].notna().sum()
    null_count = df[colonne].isna().sum()
    pourcentage = (non_null / total) * 100
    log(f"{colonne} : {non_null:,}/{total:,} ({pourcentage:.1f}%) - Vides : {null_count:,}")

print("\n" + "-"*60)
print("VERIFICATION DOUBLONS SIREN :")
print("-"*60)

# Le numero SIREN est l'identifiant unique d'une entreprise
# Aucun doublon ne doit subsister apres la transformation Silver
doublons_siren = df[df.duplicated(subset=["siren"], keep=False)].shape[0]
log(f"Nombre de doublons SIREN : {doublons_siren}")

print("\n" + "-"*60)
print("DISTRIBUTION VARIABLE CIBLE :")
print("-"*60)

# etatAdministratifUniteLegale est la variable cible du modele ML
# A = Active, C = Cessee
# Un desequilibre important entre les deux classes impactera l'entrainement
distribution = df["etatAdministratifUniteLegale"].value_counts(dropna=False)
total = len(df)
for valeur, nb in distribution.items():
    pourcentage = (nb / total) * 100
    log(f"  {valeur} : {nb:,} ({pourcentage:.1f}%)")

print("\n" + "-"*60)
print("VERIFICATION COLONNES CLES :")
print("-"*60)

# Verification des valeurs uniques pour les colonnes categoriques
# Permet de detecter des valeurs inattendues avant l'etape Gold
for colonne in ["categorieJuridiqueUniteLegale", "trancheEffectifsUniteLegale",
                "categorieEntreprise", "economieSocialeSolidaireUniteLegale"]:
    nb_uniques = df[colonne].nunique(dropna=False)
    log(f"{colonne} : {nb_uniques} valeurs uniques")

print("\n" + "-"*60)
print("APERCU DES DONNEES :")
print("-"*60)

# Affichage des 5 premieres lignes pour validation visuelle
print(df.head(5).to_string())

print("\n" + "-"*60)
print("TAILLE DU FICHIER SILVER :")
print("-"*60)

taille_mb = os.path.getsize(CHEMIN_SILVER) / (1024 * 1024)
log(f"Taille du fichier : {taille_mb:.2f} MB")

print("\n" + "="*60)
print("Analyse complete")
print("="*60)

# Suppression du log apres execution reussie
# Le log est conserve uniquement en cas d'erreur pour faciliter le diagnostic
if os.path.exists(CHEMIN_LOG):
    os.remove(CHEMIN_LOG)