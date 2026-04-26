"""
Validation de la qualite des donnees Gold
Verifie que les transformations appliquees lors de l etape Gold sont correctes
Source : sirene_entreprises_gold_latest.parquet
"""

import pandas as pd
import pyarrow.parquet as pq
import os
from datetime import datetime

CHEMIN_GOLD = "data/gold/sirene_entreprises_gold_latest.parquet"

print("=" * 60)
print("ANALYSE DE LA QUALITE - DONNEES GOLD")
print(f"Date : {datetime.now().strftime('%Y-%m-%d')}")
print("=" * 60)

if not os.path.exists(CHEMIN_GOLD):
    print("[ERREUR] Fichier Gold introuvable. Executez 05_data_preparation_gold.py.")
    exit(1)

taille_mb = os.path.getsize(CHEMIN_GOLD) / (1024 * 1024)
print(f"\n[INFO] Fichier Gold trouve : {taille_mb:.2f} MB")
print("[INFO] Chargement des donnees Gold...")

df = pq.read_table(CHEMIN_GOLD).to_pandas()

print(f"[INFO] Lignes chargees : {len(df):,}")
print(f"[INFO] Colonnes : {list(df.columns)}")

# -------------------------------------------------------
# VALEURS NULLES
# -------------------------------------------------------
print("\n" + "-" * 60)
print("VALEURS NULLES :")
print("-" * 60)
nulls = df.isnull().sum()
total_nulls = nulls.sum()
for col, n in nulls.items():
    statut = "OK" if n == 0 else "ATTENTION"
    print(f"  [{statut}] {col} : {n:,} valeurs nulles")
print(f"\n  Total valeurs nulles : {total_nulls:,}")
if total_nulls == 0:
    print("  RESULTAT : Aucune valeur nulle — donnees Gold propres")
else:
    print("  RESULTAT : Valeurs nulles detectees — verifier le script Gold")

# -------------------------------------------------------
# VARIABLE CIBLE
# -------------------------------------------------------
print("\n" + "-" * 60)
print("VARIABLE CIBLE (cible) :")
print("-" * 60)
if "cible" in df.columns:
    distribution = df["cible"].value_counts()
    total = len(df)
    nb_actives = int(distribution.get(0, 0))
    nb_cessees = int(distribution.get(1, 0))
    pct_actives = round(nb_actives / total * 100, 1)
    pct_cessees = round(nb_cessees / total * 100, 1)
    print(f"  Actives (0) : {nb_actives:,} ({pct_actives}%)")
    print(f"  Cessees (1) : {nb_cessees:,} ({pct_cessees}%)")
    valeurs_uniques = sorted(df["cible"].unique().tolist())
    if valeurs_uniques == [0, 1]:
        print("  RESULTAT : Variable cible binaire correcte (0 et 1 uniquement)")
    else:
        print(f"  RESULTAT : Valeurs inattendues detectees : {valeurs_uniques}")
else:
    print("  [ERREUR] Colonne cible introuvable")

# -------------------------------------------------------
# PLAGE DES VALEURS NORMALISEES
# -------------------------------------------------------
print("\n" + "-" * 60)
print("PLAGE DES VALEURS (colonnes normalisees entre 0 et 1) :")
print("-" * 60)
colonnes_normalisees = [
    "anciennete_annees",
    "duree_periode_actuelle_annees",
    "nombrePeriodesUniteLegale"
]
for col in colonnes_normalisees:
    if col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        statut = "OK" if 0 <= min_val and max_val <= 1 else "ATTENTION"
        print(f"  [{statut}] {col} : min={min_val:.4f} | max={max_val:.4f}")

# -------------------------------------------------------
# ENCODAGES ORDINAUX
# -------------------------------------------------------
print("\n" + "-" * 60)
print("ENCODAGES ORDINAUX :")
print("-" * 60)
colonnes_ordinales = {
    "trancheEffectifsUniteLegale": (0, 15),
    "categorieEntreprise": (0, 4),
    "economieSocialeSolidaireUniteLegale": (-1, 1)
}
for col, (min_attendu, max_attendu) in colonnes_ordinales.items():
    if col in df.columns:
        min_val = int(df[col].min())
        max_val = int(df[col].max())
        statut = "OK" if min_val >= min_attendu and max_val <= max_attendu else "ATTENTION"
        print(f"  [{statut}] {col} : min={min_val} | max={max_val} | attendu=[{min_attendu},{max_attendu}]")

# -------------------------------------------------------
# ENCODAGES PAR FREQUENCE
# -------------------------------------------------------
print("\n" + "-" * 60)
print("ENCODAGES PAR FREQUENCE :")
print("-" * 60)
colonnes_frequence = [
    "categorieJuridiqueUniteLegale",
    "activitePrincipaleUniteLegale"
]
for col in colonnes_frequence:
    if col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        statut = "OK" if 0 <= min_val and max_val <= 1 else "ATTENTION"
        print(f"  [{statut}] {col} : min={min_val:.6f} | max={max_val:.6f}")

# -------------------------------------------------------
# TAILLE DU FICHIER
# -------------------------------------------------------
print("\n" + "-" * 60)
print("TAILLE DU FICHIER GOLD :")
print("-" * 60)
print(f"  Taille : {taille_mb:.2f} MB")
print(f"  Lignes : {len(df):,}")
print(f"  Colonnes : {len(df.columns)}")

print("\n" + "=" * 60)
print("Analyse complete")
print(f"Date fin : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("=" * 60)