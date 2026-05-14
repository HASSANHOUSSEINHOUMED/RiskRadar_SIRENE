"""
Script de verification de la qualite des donnees SIRENE - ETAPE BRONZE
Objectif : valider l'integrite du fichier telecharge avant transformation Silver
Source des donnees : data/raw/sirene_entreprises_bronze_latest.parquet
"""

import pyarrow.parquet as pq
import os

chemin_parquet = "data/raw/sirene_entreprises_bronze_latest.parquet"

if not os.path.exists(chemin_parquet):
    print(f"ERREUR : Fichier {chemin_parquet} non trouve")
    exit()

print("\n" + "="*60)
print("ANALYSE DE LA QUALITE - DONNEES BRONZE COMPLETES")
print("="*60)

# Lecture des metadonnees uniquement - pas de chargement en RAM
metadata = pq.read_metadata(chemin_parquet)
schema = pq.read_schema(chemin_parquet)

nb_lignes = metadata.num_rows
nb_colonnes = len(schema)
taille_mb = os.path.getsize(chemin_parquet) / (1024 * 1024)

print(f"\nNombre total de lignes : {nb_lignes:,}")
print(f"Nombre total de colonnes : {nb_colonnes}")
print(f"Taille du fichier : {taille_mb:.2f} MB")

print("\n" + "-"*60)
print("COLONNES DISPONIBLES :")
print("-"*60)
for i, field in enumerate(schema):
    print(f"  {i+1:02d}. {field.name} ({field.type})")

# Chargement uniquement des colonnes necessaires pour la qualite
print("\n" + "-"*60)
print("COMPLETUDE COLONNES CLES :")
print("-"*60)

colonnes_cles = [
    "siren",
    "etatAdministratifUniteLegale",
    "categorieJuridiqueUniteLegale",
    "activitePrincipaleUniteLegale",
    "trancheEffectifsUniteLegale",
    "categorieEntreprise",
    "economieSocialeSolidaireUniteLegale",
    "dateCreationUniteLegale",
    "nombrePeriodesUniteLegale"
]

# Filtrer uniquement les colonnes qui existent dans le fichier
colonnes_existantes = [c for c in colonnes_cles if c in schema.names]
table = pq.read_table(chemin_parquet, columns=colonnes_existantes)
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
print("DISTRIBUTION ETAT ADMINISTRATIF :")
print("-"*60)
if "etatAdministratifUniteLegale" in df.columns:
    distribution = df["etatAdministratifUniteLegale"].value_counts(dropna=False)
    total = len(df)
    for valeur, nb in distribution.items():
        pourcentage = (nb / total) * 100
        print(f"  {valeur} : {nb:,} ({pourcentage:.1f}%)")

print("\n" + "="*60)
print("Analyse complete")
print("="*60)