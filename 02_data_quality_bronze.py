"""
Script de verification de la qualite des donnees SIRENE - ETAPE BRONZE
Objectif : valider l'integrite du fichier telecharge avant transformation Silver
Source des donnees : data/raw/sirene_entreprises_bronze_latest.parquet
"""

import pandas as pd
import os

# Chemin vers le fichier bronze telecharge lors de l'etape precedente
chemin_parquet = "data/raw/sirene_entreprises_bronze_latest.parquet"

# Verification de l'existence du fichier avant tout traitement
if not os.path.exists(chemin_parquet):
    print(f"ERREUR : Fichier {chemin_parquet} non trouve")
    exit()

# Chargement du fichier Parquet en memoire via pandas
print("Chargement des donnees bronze...")
df = pd.read_parquet(chemin_parquet)

print("\n" + "="*60)
print("ANALYSE DE LA QUALITE - DONNEES BRONZE COMPLETES")
print("="*60)

# Dimensions generales du jeu de donnees
print(f"\nNombre total de lignes : {len(df):,}")
print(f"Nombre total de colonnes : {len(df.columns)}")

print("\n" + "-"*60)
print("COMPLETUDE PAR COLONNE :")
print("-"*60)

# Calcul du taux de remplissage pour chaque colonne
# Permet d'identifier les colonnes trop vides avant de les utiliser en Silver
for colonne in df.columns:
    total = len(df)
    non_null = df[colonne].notna().sum()
    null_count = df[colonne].isna().sum()
    pourcentage = (non_null / total) * 100
    print(f"\n{colonne}:")
    print(f"  Remplis : {non_null:,}/{total:,} ({pourcentage:.1f}%)")
    print(f"  Vides (NULL) : {null_count:,}")

print("\n" + "-"*60)
print("SYNTHESE PAR CATEGORIE :")
print("-"*60)

# Classification des colonnes selon leur taux de completude
# Seuils choisis : >=80% bien remplie, 20-80% partielle, <20% quasi-vide
colonnes_bien_remplies = []
colonnes_partiellement_remplies = []
colonnes_vides = []

for colonne in df.columns:
    pourcentage = (df[colonne].notna().sum() / len(df)) * 100
    if pourcentage >= 80:
        colonnes_bien_remplies.append((colonne, pourcentage))
    elif pourcentage >= 20:
        colonnes_partiellement_remplies.append((colonne, pourcentage))
    else:
        colonnes_vides.append((colonne, pourcentage))

print(f"\nBien remplies (>=80%) : {len(colonnes_bien_remplies)}")
for col, pct in colonnes_bien_remplies:
    print(f"  - {col} : {pct:.1f}%")

print(f"\nPartiellement remplies (20-80%) : {len(colonnes_partiellement_remplies)}")
for col, pct in colonnes_partiellement_remplies:
    print(f"  - {col} : {pct:.1f}%")

print(f"\nVides (<20%) : {len(colonnes_vides)}")
for col, pct in colonnes_vides:
    print(f"  - {col} : {pct:.1f}%")

print("\n" + "-"*60)
print("VERIFICATION DOUBLONS SIREN :")
print("-"*60)

# Le numero SIREN est l'identifiant unique d'une entreprise
# Un doublon indique une anomalie dans les donnees sources
doublons_siren = df[df.duplicated(subset=["siren"], keep=False)].shape[0]
print(f"Nombre de doublons SIREN : {doublons_siren}")

print("\n" + "-"*60)
print("DISTRIBUTION ETAT ADMINISTRATIF :")
print("-"*60)

# etatAdministratifUniteLegale est la variable cible du projet
# A = Active, C = Cessée - c'est cette colonne qui servira a construire la variable cible ML
if "etatAdministratifUniteLegale" in df.columns:
    distribution = df["etatAdministratifUniteLegale"].value_counts(dropna=False)
    print(distribution)
else:
    print("Colonne etatAdministratifUniteLegale non trouvee")

print("\n" + "-"*60)
print("TAILLE DU FICHIER PARQUET :")
print("-"*60)

# Verification de la taille physique du fichier sur disque
taille_mb = os.path.getsize(chemin_parquet) / (1024 * 1024)
print(f"Taille du fichier : {taille_mb:.2f} MB")

print("\n" + "="*60)
print("Analyse complete")
print("="*60)