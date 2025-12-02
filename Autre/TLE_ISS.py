import csv
import math
from datetime import datetime, timedelta, timezone

import requests

# ==========
# CONSTANTES
# ==========

SPACETRACK_USER = "leovasseur30@gmail.com"
SPACETRACK_PASS = "dytnym-kehryz-6xedKo"

NORAD_ID = 25544          # ISS
SAT_NAME = "ISS"
OUTPUT_CSV = "iss_last_2000_tles_spacetrack.csv"

BASE_URL = "https://www.space-track.org"

LOGIN_URL = BASE_URL + "/ajaxauth/login"
TLE_URL = (
    BASE_URL
    + "/basicspacedata/query/"
    "class/tle/"
    f"NORAD_CAT_ID/{NORAD_ID}/"
    "orderby/EPOCH%20desc/"
    "limit/2000/"
    "format/tle"
)


def parse_tle_epoch(epoch_str: str) -> datetime:
    """
    Convertit l'époque TLE (YYDDD.DDDDD) en datetime UTC.
    """
    epoch_str = epoch_str.strip()
    yy = int(epoch_str[:2])
    doy = float(epoch_str[2:])
    year = 1900 + yy if yy >= 57 else 2000 + yy

    day_int = int(math.floor(doy))
    frac = doy - day_int

    dt0 = datetime(year, 1, 1, tzinfo=timezone.utc)
    dt = dt0 + timedelta(days=day_int - 1, seconds=frac * 86400)
    return dt


def fetch_last_iss_tles_spacetrack():
    """
    Récupère les 20 derniers TLE de l'ISS (NORAD 25544) via Space-Track
    et retourne une liste de tuples (epoch_datetime, name, line1, line2).
    """
    with requests.Session() as s:
        # 1) Login
        print("Connexion à Space-Track...")
        payload = {
            "identity": SPACETRACK_USER,
            "password": SPACETRACK_PASS,
        }
        r_login = s.post(LOGIN_URL, data=payload, timeout=30)
        r_login.raise_for_status()

        # Si Space-Track renvoie une page HTML d'erreur
        if "You are now logged in" not in r_login.text and r_login.status_code != 200:
            print("Réponse login Space-Track inhabituelle :")
            print(r_login.text[:500])
            raise RuntimeError("Login Space-Track peut-être invalide (login/mot de passe ?)")

        print("Connecté, récupération des TLE...")

        # 2) Requête TLE
        r_tle = s.get(TLE_URL, timeout=30)
        r_tle.raise_for_status()

        text = r_tle.text.strip()
        if not text:
            print("Réponse brute Space-Track vide :")
            print(f"Status code: {r_tle.status_code}")
            print(f"Headers: {r_tle.headers}")
            raise RuntimeError("Réponse vide de Space-Track (pas de TLE).")

        lines = [l.strip() for l in text.splitlines() if l.strip()]

        if len(lines) % 2 != 0:
            raise RuntimeError("Nombre de lignes TLE impair, format inattendu (Space-Track renvoie normalement 2 lignes par TLE).")

        tles = []
        for i in range(0, len(lines), 2):
            l1 = lines[i]
            l2 = lines[i + 1]

            if not (l1.startswith("1 ") and l2.startswith("2 ")):
                print(f"Bloc ignoré (pas TLE valide) :\n{l1}\n{l2}")
                continue

            try:
                catnum = int(l1[2:7])
            except ValueError:
                print(f"Impossible de lire le NORAD dans '{l1}'")
                continue

            if catnum != NORAD_ID:
                # sécurité
                continue

            epoch_field = l1[18:32]
            epoch_dt = parse_tle_epoch(epoch_field)

            tles.append((epoch_dt, SAT_NAME, l1, l2))

        if not tles:
            print("Contenu texte Space-Track (début) :")
            print(text[:500])
            raise RuntimeError("Aucun TLE ISS (25544) trouvé dans la réponse Space-Track.")

        # normalement déjà triés par EPOCH desc, mais on sécurise
        tles.sort(key=lambda x: x[0], reverse=True)
        # garder max 2000 TLE
        tles = tles[:2000]

        print(f"{len(tles)} TLE récupérés pour ISS (NORAD {NORAD_ID}).")
        return tles


def save_tles_to_csv(tles):
    """
    Enregistre les TLE dans OUTPUT_CSV au format :
    epoch;name;line1;line2
    """
    print(f"Écriture des TLE dans {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["epoch", "name", "line1", "line2"])
        for epoch_dt, name, l1, l2 in tles:
            w.writerow([epoch_dt.isoformat(), name, l1, l2])
    print("Terminé.")


if __name__ == "__main__":
    tles = fetch_last_iss_tles_spacetrack()
    save_tles_to_csv(tles)

    # Aperçu dans le terminal
    for i, (epoch, name, l1, l2) in enumerate(tles, start=1):
        print(f"\nTLE {i}/2000  epoch={epoch}")
        print(name)
        print(l1)
        print(l2)
