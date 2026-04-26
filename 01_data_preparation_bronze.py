"""
Script de preparation des donnees SIRENE - ETAPE BRONZE
Objectif : telecharger le fichier stock mensuel des unites legales
Source : data.gouv.fr / INSEE SIRENE
https://www.data.gouv.fr/fr/datasets/base-sirene-des-entreprises-et-de-leurs-etablissements-siren-siret/
"""

import os
import requests
import json
from datetime import datetime

# Date du jour utilisee pour nommer les fichiers et gerer les checkpoints mensuels
DATE_EXECUTION = datetime.now().strftime("%Y-%m-%d")
MOIS_EXECUTION = DATE_EXECUTION[:7]

# Chemins des fichiers de sortie et de suivi
DOSSIER_RAW = "data/raw"
CHEMIN_FINAL = os.path.join(DOSSIER_RAW, f"sirene_entreprises_bronze_{DATE_EXECUTION}.parquet")
CHEMIN_LATEST = os.path.join(DOSSIER_RAW, "sirene_entreprises_bronze_latest.parquet")
CHEMIN_LOG = os.path.join(DOSSIER_RAW, f"recuperation_log_{DATE_EXECUTION}.txt")
CHEMIN_ETAT = os.path.join(DOSSIER_RAW, "etat_recuperation.json")

# URL du fichier Parquet stock mensuel publie par l'INSEE sur data.gouv.fr
# Ce fichier contient toutes les unites legales actives et cessees depuis 1973
URL_STOCK = "https://object.files.data.gouv.fr/data-pipeline-open/siren/stock/StockUniteLegale_utf8.parquet"

# Taille des blocs de telechargement : 8 Mo par chunk pour equilibrer vitesse et memoire
TAILLE_CHUNK_DOWNLOAD = 8 * 1024 * 1024

# Creation du dossier de destination si inexistant
os.makedirs(DOSSIER_RAW, exist_ok=True)


def log(message):
    # Ecriture horodatee dans le terminal et dans le fichier log
    # Le log est conserve en cas d'interruption pour faciliter le diagnostic
    horodatage = datetime.now().strftime("%H:%M:%S")
    ligne = f"[{horodatage}] {message}"
    print(ligne)
    with open(CHEMIN_LOG, "a", encoding="utf-8") as f:
        f.write(ligne + "\n")


def charger_etat():
    # Lecture de l'etat de la recuperation precedente
    # Si le fichier d'etat appartient a un mois different, on repart de zero
    # Cela permet l'automatisation mensuelle sans intervention manuelle
    if os.path.exists(CHEMIN_ETAT):
        with open(CHEMIN_ETAT, "r", encoding="utf-8") as f:
            etat = json.load(f)
        if etat.get("mois") == MOIS_EXECUTION:
            return etat
        else:
            log(f"Etat d'un mois precedent ({etat.get('mois')}) ignore - nouvelle recuperation mensuelle")
            os.remove(CHEMIN_ETAT)
    return {"mois": MOIS_EXECUTION, "telechargement_complet": False}


def sauvegarder_etat(etat):
    # Persistance de l'etat apres chaque etape critique
    # Permet la reprise automatique en cas d'interruption
    with open(CHEMIN_ETAT, "w", encoding="utf-8") as f:
        json.dump(etat, f, ensure_ascii=False)


def telecharger_fichier(etat):
    if etat.get("telechargement_complet") and os.path.exists(CHEMIN_FINAL):
        log("Fichier PARQUET deja telecharge - etape ignoree")
        return True

    # Nombre de tentatives avant d abandonner
    NB_TENTATIVES = 3
    ATTENTE_SECONDES = 60

    for tentative in range(1, NB_TENTATIVES + 1):
        log(f"Tentative {tentative}/{NB_TENTATIVES} — Telechargement depuis : {URL_STOCK}")

        octets_deja = 0
        headers = {}
        if os.path.exists(CHEMIN_FINAL):
            octets_deja = os.path.getsize(CHEMIN_FINAL)
            headers["Range"] = f"bytes={octets_deja}-"
            log(f"Reprise du telechargement a partir de {octets_deja / (1024**3):.2f} Go")

        try:
            response = requests.get(URL_STOCK, headers=headers, stream=True, timeout=60)

            if response.status_code not in (200, 206):
                log(f"Erreur HTTP : {response.status_code}")
                raise Exception(f"Code HTTP inattendu : {response.status_code}")

            taille_totale = int(response.headers.get("content-length", 0)) + octets_deja
            log(f"Taille totale estimee : {taille_totale / (1024**3):.2f} Go")

            mode = "ab" if octets_deja > 0 else "wb"
            octets_recus = octets_deja

            with open(CHEMIN_FINAL, mode) as f:
                for chunk in response.iter_content(chunk_size=TAILLE_CHUNK_DOWNLOAD):
                    if chunk:
                        f.write(chunk)
                        octets_recus += len(chunk)
                        if taille_totale > 0:
                            pourcentage = (octets_recus / taille_totale) * 100
                            print(f"\r  Progression : {octets_recus / (1024**3):.2f} Go / {taille_totale / (1024**3):.2f} Go ({pourcentage:.1f}%)", end="", flush=True)

            print()
            log("Telechargement termine")

            import shutil
            shutil.copy2(CHEMIN_FINAL, CHEMIN_LATEST)
            log(f"Fichier final : {CHEMIN_FINAL}")
            log(f"Fichier latest : {CHEMIN_LATEST}")

            etat["telechargement_complet"] = True
            sauvegarder_etat(etat)
            return True

        except Exception as e:
            log(f"Erreur tentative {tentative}/{NB_TENTATIVES} : {str(e)}")
            if tentative < NB_TENTATIVES:
                log(f"Nouvelle tentative dans {ATTENTE_SECONDES} secondes...")
                import time
                time.sleep(ATTENTE_SECONDES)
            else:
                log("Nombre maximum de tentatives atteint")
                if os.path.exists(CHEMIN_LATEST):
                    log("Utilisation du fichier du mois precedent")
                    return True
                else:
                    log("Aucun fichier precedent disponible — pipeline interrompu")
                    return False

def nettoyer():
    # Suppression des fichiers de suivi apres recuperation reussie
    # Le log est egalement supprime pour repartir proprement au prochain mois
    if os.path.exists(CHEMIN_ETAT):
        os.remove(CHEMIN_ETAT)
    log("Nettoyage termine - prochaine recuperation repartira proprement")
    if os.path.exists(CHEMIN_LOG):
        os.remove(CHEMIN_LOG)


print("\n" + "="*60)
print("RECUPERATION DES DONNEES SIRENE - BRONZE")
print(f"Source : data.gouv.fr | Date : {DATE_EXECUTION}")
print("="*60)

log("Demarrage du script")

# Chargement de l'etat pour determiner si on reprend ou si on repart de zero
etat = charger_etat()

succes = telecharger_fichier(etat)
if not succes:
    exit()

nettoyer()

log("RECUPERATION BRONZE COMPLETE")

print("\n" + "="*60)
print(f"Date fin : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("="*60)