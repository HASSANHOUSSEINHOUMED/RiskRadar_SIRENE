"""
Script de preparation des donnees SIRENE - ETAPE GOLD
Objectif : encoder, normaliser et construire la variable cible binaire
Source : data/silver/sirene_entreprises_silver_latest.parquet
Sortie : data/gold/sirene_entreprises_gold_YYYY-MM-DD.parquet
"""

import os
import shutil
import pandas as pd
from datetime import datetime

DATE_EXECUTION = datetime.now().strftime("%Y-%m-%d")
DATE_REFERENCE = pd.Timestamp(DATE_EXECUTION)

# Chemins des fichiers source et de sortie
DOSSIER_GOLD = "data/gold"
CHEMIN_SILVER = "data/silver/sirene_entreprises_silver_latest.parquet"
CHEMIN_FINAL = os.path.join(DOSSIER_GOLD, f"sirene_entreprises_gold_{DATE_EXECUTION}.parquet")
CHEMIN_LATEST = os.path.join(DOSSIER_GOLD, "sirene_entreprises_gold_latest.parquet")
CHEMIN_LOG = os.path.join(DOSSIER_GOLD, f"gold_log_{DATE_EXECUTION}.txt")

os.makedirs(DOSSIER_GOLD, exist_ok=True)


def log(message):
    # Ecriture horodatee dans le terminal et dans le fichier log
    horodatage = datetime.now().strftime("%H:%M:%S")
    ligne = f"[{horodatage}] {message}"
    print(ligne)
    with open(CHEMIN_LOG, "a", encoding="utf-8") as f:
        f.write(ligne + "\n")


def verifier_source():
    # Verification de l'existence du fichier Silver avant tout traitement
    if not os.path.exists(CHEMIN_SILVER):
        log(f"ERREUR : Fichier Silver non trouve : {CHEMIN_SILVER}")
        return False
    taille_mb = os.path.getsize(CHEMIN_SILVER) / (1024 * 1024)
    log(f"Fichier Silver trouve : {taille_mb:.2f} MB")
    return True


def construire_variable_cible(df):
    # Variable cible binaire : 0 = entreprise active, 1 = entreprise cessee
    # C'est la variable que les modeles ML doivent predire
    df["cible"] = (df["etatAdministratifUniteLegale"] == "C").astype(int)
    log(f"Variable cible construite - Actives (0) : {(df['cible'] == 0).sum():,} | Cessees (1) : {(df['cible'] == 1).sum():,}")
    return df


def construire_features_temporelles(df):
    # Calcul de l'anciennete de l'entreprise en annees a partir de la date de creation
    # Une entreprise ancienne a un profil de risque different d'une jeune entreprise
    df["dateCreationUniteLegale"] = pd.to_datetime(df["dateCreationUniteLegale"], errors="coerce")
    df["anciennete_annees"] = (DATE_REFERENCE - df["dateCreationUniteLegale"]).dt.days / 365.25
    df["anciennete_annees"] = df["anciennete_annees"].clip(lower=0)

    # Calcul de la duree de la derniere periode en annees
    # Reflète la stabilite recente de l'entreprise
    df["dateDebut"] = pd.to_datetime(df["dateDebut"], errors="coerce")
    df["duree_periode_actuelle_annees"] = (DATE_REFERENCE - df["dateDebut"]).dt.days / 365.25
    df["duree_periode_actuelle_annees"] = df["duree_periode_actuelle_annees"].clip(lower=0)

    # Suppression des colonnes dates brutes remplacees par les features calculees
    df = df.drop(columns=["dateCreationUniteLegale", "dateDebut"])
    log("Features temporelles construites : anciennete_annees, duree_periode_actuelle_annees")
    return df


def encoder_colonnes_categorielles(df):
    # Encodage ordinal de trancheEffectifsUniteLegale
    # Les tranches sont ordonnees de la plus petite a la plus grande
    ordre_tranches = {
        "NN": 0, "00": 1, "01": 2, "02": 3, "03": 4,
        "11": 5, "12": 6, "21": 7, "22": 8, "31": 9,
        "32": 10, "41": 11, "42": 12, "51": 13, "52": 14, "53": 15
    }
    df["trancheEffectifsUniteLegale"] = df["trancheEffectifsUniteLegale"].map(ordre_tranches).fillna(0).astype(int)
    log("Encodage ordinal applique : trancheEffectifsUniteLegale")

    # Encodage ordinal de categorieEntreprise
    # Ordre croissant par taille : TPE < PME < ETI < GE
    ordre_categorie = {"TPE": 1, "PME": 2, "ETI": 3, "GE": 4}
    df["categorieEntreprise"] = df["categorieEntreprise"].map(ordre_categorie).fillna(0).astype(int)
    log("Encodage ordinal applique : categorieEntreprise")

    # Encodage binaire de economieSocialeSolidaireUniteLegale
    # O = Oui (ESS), N = Non, NaN = Inconnu (-1)
    df["economieSocialeSolidaireUniteLegale"] = df["economieSocialeSolidaireUniteLegale"].map(
        {"O": 1, "N": 0}
    ).fillna(-1).astype(int)
    log("Encodage binaire applique : economieSocialeSolidaireUniteLegale")

    # Encodage par frequence de categorieJuridiqueUniteLegale
    # Les categories rares ont une frequence basse, les communes une frequence haute
    freq = df["categorieJuridiqueUniteLegale"].value_counts(normalize=True)
    df["categorieJuridiqueUniteLegale"] = df["categorieJuridiqueUniteLegale"].map(freq).astype(float)
    log("Encodage par frequence applique : categorieJuridiqueUniteLegale")

    # Encodage par frequence de activitePrincipaleUniteLegale
    # Le code NAF a trop de valeurs uniques pour un encodage one-hot
    freq_naf = df["activitePrincipaleUniteLegale"].value_counts(normalize=True)
    df["activitePrincipaleUniteLegale"] = df["activitePrincipaleUniteLegale"].map(freq_naf).fillna(0).astype(float)
    log("Encodage par frequence applique : activitePrincipaleUniteLegale")

    return df


def normaliser_colonnes_numeriques(df):
    # Normalisation min-max des colonnes numeriques continues
    # Permet aux modeles ML de traiter toutes les features sur la meme echelle
    colonnes_a_normaliser = [
        "anciennete_annees",
        "duree_periode_actuelle_annees",
        "nombrePeriodesUniteLegale"
    ]

    for colonne in colonnes_a_normaliser:
        min_val = df[colonne].min()
        max_val = df[colonne].max()
        if max_val > min_val:
            df[colonne] = (df[colonne] - min_val) / (max_val - min_val)
        else:
            df[colonne] = 0
        log(f"Normalisation min-max appliquee : {colonne}")

    return df


def traiter_valeurs_nulles(df):
    # Remplacement des valeurs nulles par la mediane de chaque colonne
    # La mediane est privilegiee par rapport a la moyenne car elle est robuste aux valeurs extremes
    nb_nulls_avant = df.isnull().sum().sum()
    for colonne in df.columns:
        if df[colonne].isnull().sum() > 0:
            mediane = df[colonne].median()
            df[colonne] = df[colonne].fillna(mediane)
            log(f"Nulls remplaces par mediane ({mediane:.4f}) : {colonne}")
    nb_nulls_apres = df.isnull().sum().sum()
    log(f"Valeurs nulles avant traitement : {nb_nulls_avant:,} | Apres : {nb_nulls_apres:,}")
    return df


def supprimer_colonnes_inutiles(df):
    # Suppression des colonnes non utilisees pour l'entrainement ML
    colonnes_a_supprimer = ["etatAdministratifUniteLegale", "siren"]
    df = df.drop(columns=colonnes_a_supprimer)
    log(f"Colonnes supprimees apres encodage : {colonnes_a_supprimer}")
    return df


print("\n" + "="*60)
print("PREPARATION DES DONNEES SIRENE - GOLD")
print(f"Date : {DATE_EXECUTION}")
print("="*60)

log("Demarrage du script Gold")

if not verifier_source():
    exit()

log("Chargement du fichier Silver...")
df = pd.read_parquet(CHEMIN_SILVER)
log(f"Lignes chargees : {len(df):,}")

# Construction de la variable cible binaire
df = construire_variable_cible(df)

# Construction des features temporelles
df = construire_features_temporelles(df)

# Encodage des colonnes categorielles
df = encoder_colonnes_categorielles(df)

# Normalisation des colonnes numeriques
df = normaliser_colonnes_numeriques(df)

# Traitement des valeurs nulles restantes apres normalisation
df = traiter_valeurs_nulles(df)

# Suppression des colonnes non necessaires pour le ML
df = supprimer_colonnes_inutiles(df)

# Verification finale du DataFrame Gold
log(f"Colonnes finales Gold : {list(df.columns)}")
log(f"Lignes finales : {len(df):,}")
log(f"Valeurs nulles restantes : {df.isnull().sum().sum()}")

# Sauvegarde du fichier Gold
log("Sauvegarde du fichier Gold...")
df.to_parquet(CHEMIN_FINAL, engine="pyarrow", index=False)
shutil.copy2(CHEMIN_FINAL, CHEMIN_LATEST)

taille_mb = os.path.getsize(CHEMIN_FINAL) / (1024 * 1024)
log(f"Fichier Gold sauvegarde : {taille_mb:.2f} MB")
log(f"Fichier latest mis a jour : {CHEMIN_LATEST}")

log("PREPARATION GOLD COMPLETE")

# Suppression du log apres execution reussie
if os.path.exists(CHEMIN_LOG):
    os.remove(CHEMIN_LOG)

print("\n" + "="*60)
print(f"Date fin : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("="*60)