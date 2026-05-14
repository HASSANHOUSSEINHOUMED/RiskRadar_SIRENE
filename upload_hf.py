"""
Script de mise à jour des données sur Hugging Face
Permet de mettre à jour le dataset avec les nouvelles données mensuelles SIRENE
Source : data/silver/ et data/modeles/
"""

from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()

HF_REPO_ID = os.getenv("HF_REPO_ID")

if not HF_REPO_ID:
    print("[ERREUR] HF_REPO_ID manquant dans le fichier .env")
    exit(1)

fichiers = [
    ("data/silver/sirene_entreprises_silver_latest.parquet", "sirene_entreprises_silver_latest.parquet"),
    ("data/modeles/performances.json", "performances.json"),
    ("data/modeles/modele_xgboost.pkl", "modele_xgboost.pkl"),
    ("data/modeles/modele_random_forest.pkl", "modele_random_forest.pkl"),
    ("data/modeles/modele_logistic_regression.pkl", "modele_logistic_regression.pkl"),
]

print("=" * 60)
print("MISE A JOUR HUGGING FACE - RISKRADAR SIRENE")
print(f"Dataset : {HF_REPO_ID}")
print("=" * 60)

api = HfApi()

for chemin_local, nom_hf in fichiers:
    if not os.path.exists(chemin_local):
        print(f"[ATTENTION] Fichier introuvable : {chemin_local}")
        continue
    taille_mb = os.path.getsize(chemin_local) / (1024 * 1024)
    print(f"\nUpload : {nom_hf} ({taille_mb:.2f} MB)...")
    api.upload_file(
        path_or_fileobj=chemin_local,
        path_in_repo=nom_hf,
        repo_id=HF_REPO_ID,
        repo_type="dataset"
    )
    print(f"OK : {nom_hf}")

print("\n" + "=" * 60)
print("Mise à jour Hugging Face terminée")
print("=" * 60)