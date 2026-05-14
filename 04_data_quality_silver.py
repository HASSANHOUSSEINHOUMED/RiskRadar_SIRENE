"""
Script de verification de la qualite des donnees SIRENE - ETAPE SILVER
Objectif : valider l'integrite du fichier Silver avant transformation Gold
Source : data/silver/sirene_entreprises_silver_latest.parquet
"""

import pyarrow.parquet as pq
import os
from datetime import datetime

DATE_EXECUTION = datetime.now().strftime("%Y-%m-%d")
CHEMIN_SILVER = "data/silver/sirene_entreprises_silver_latest.parquet"

if not os.path.exists(CHEMIN_SILVER):
    print(f"ERREUR : Fichier {CHEMIN_SILVER} non trouve")
    exit()

print("\n" + "="*60)
print("ANALYSE DE LA QUALITE - DONNEES SILVER")
print("="*60)

# Lecture des metadonnees uniquement
metadata = pq.read_metadata(CHEMIN_SILVER)
schema = pq.read_schema(CHEMIN_SILVER)
taille_mb = os.path.getsize(CHEMIN_SILVER) / (1024 * 1024)

print(f"\nNombre total de lignes : {metadata.num_rows:,}")
print(f"Nombre total de colonnes : {len(schema)}")
print(f"Colonnes : {schema.names}")
print(f"Taille du fichier : {taille_mb:.2f} MB")

# Chargement optimise avec selection de colonnes
print("\n" + "-"*60)
print("COMPLETUDE PAR COLONNE :")
print("-"*60)

table = pq.read_table(CHEMIN_SILVER)
df = table.to_pandas()

for colonne in df.columns:
    total = len(df)
    non_null = df[colonne].notna().sum()
    null_count = df[colonne].isna().sum()
    pourcentage = (non_null / total) * 100
    print(f"  {colonne} : {pourcentage:.1f}% remplie ({null_count:,} vides)")

print("\n" + "-"*60)
print("VERIFICATION DOUBLONS SIREN :")
print("-"*60)
doublons_siren = df[df.duplicated(subset=["siren"], keep=False)].shape[0]
print(f"Nombre de doublons SIREN : {doublons_siren:,}")

print("\n" + "-"*60)
print("DISTRIBUTION VARIABLE CIBLE :")
print("-"*60)
distribution = df["etatAdministratifUniteLegale"].value_counts(dropna=False)
total = len(df)
for valeur, nb in distribution.items():
    pourcentage = (nb / total) * 100
    print(f"  {valeur} : {nb:,} ({pourcentage:.1f}%)")

print("\n" + "-"*60)
print("VERIFICATION COLONNES CLES :")
print("-"*60)
for colonne in ["categorieJuridiqueUniteLegale", "trancheEffectifsUniteLegale",
                "categorieEntreprise", "economieSocialeSolidaireUniteLegale"]:
    if colonne in df.columns:
        nb_uniques = df[colonne].nunique(dropna=False)
        print(f"  {colonne} : {nb_uniques} valeurs uniques")

print("\n" + "-"*60)
print("APERCU DES DONNEES :")
print("-"*60)
print(df.head(3).to_string())

print("\n" + "="*60)
print("Analyse complete")
print("="*60)