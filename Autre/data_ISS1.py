import csv
import math
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dateutil import parser as dateparser

from skyfield.api import Loader, EarthSatellite
from astroquery.jplhorizons import Horizons
from astropy.time import Time
from requests.exceptions import HTTPError

# =========================
# 1) Constantes
# =========================

CSV_TLE_FILE = "iss_last_20_tles_spacetrack.csv"  # fichier epoch;name;line1;line2 (peut être ajusté)
HORIZONS_ID = "-125544"  # ISS
STEP_MINUTES = 5
EXTRA_HOURS_LAST = 6
OUTPUT_DATASET = "dataset_iss_sgp4_vs_horizons.csv"

HORIZONS_CHUNK_SIZE = 20   # taille fixe pour les requêtes Horizons
MAX_RETRIES = 3          # nb de tentatives par chunk avant NaN
RETRY_SLEEP_SECONDS = 3  # pause entre tentatives

load = Loader(".")
ts = load.timescale()


# =========================
# 2) Chargement des TLE
# =========================

def load_tles_from_csv(path):
    """
    Lit un CSV epoch;name;line1;line2 et renvoie une liste triée :
    [(epoch_dt, name, l1, l2), ...]
    """
    df = pd.read_csv(path, sep=";")
    tles = []
    for _, row in df.iterrows():
        epoch_str = row["epoch"]
        epoch_dt = dateparser.parse(epoch_str)  # gère ISO automatiquement
        name = str(row["name"])
        l1 = str(row["line1"])
        l2 = str(row["line2"])
        tles.append((epoch_dt, name, l1, l2))

    # tri par epoch croissant
    tles.sort(key=lambda x: x[0])
    return tles


# =========================
# 3) Trajectoire SGP4
# =========================

def generate_sgp4_trajectory(tles, step_minutes=STEP_MINUTES,
                             extra_hours_last=EXTRA_HOURS_LAST):
    """
    tles: liste (epoch_dt, name, l1, l2) triée.
    Retourne:
        times_dt      : liste de datetime UTC
        sgp4_pos      : np.ndarray (N,3) positions (km)
        tle_index     : liste d'indices TLE (0,1,2,…) pour chaque point
        dt_since_tle  : np.ndarray (N,) delta t en secondes depuis l'epoch du TLE
        tle_params    : liste de dicts par TLE avec les paramètres orbitaux clés
    """
    all_times = []
    all_positions = []
    all_tle_idx = []
    all_dt_since = []
    tle_params = []

    def extract_tle_params(satellite: EarthSatellite):
        """Extrait les paramètres orbitaux utiles depuis le modèle SGP4."""
        model = satellite.model
        # Mean motion (rev/jour) à partir de no_kozai (rad/min)
        mean_motion_rev_day = (model.no_kozai * 1440.0) / (2 * math.pi)
        # Dérivée de la mean motion (rev/jour^2) à partir de ndot (rad/min^2)
        mean_motion_derivative_rev_day2 = (model.ndot * (1440.0 ** 2)) / (2 * math.pi)

        # Estimation vitesse orbitale (km/s) en supposant orbite quasi circulaire
        mu_earth = 398600.4418  # km^3 / s^2
        n_rad_s = model.no_kozai / 60.0
        a_km = (mu_earth / (n_rad_s ** 2)) ** (1 / 3) if n_rad_s != 0 else np.nan
        orbital_speed_km_s = n_rad_s * a_km if n_rad_s != 0 else np.nan

        # Dérive approximative du demi-grand axe (km/jour) liée à la traînée
        n_dot_rad_s2 = model.ndot / (60.0 ** 2)
        if n_rad_s != 0:
            da_dt_km_s = -(2.0 / 3.0) * (a_km / n_rad_s) * n_dot_rad_s2
            altitude_drift_km_per_day = da_dt_km_s * 86400.0
        else:
            altitude_drift_km_per_day = np.nan

        return {
            "mean_motion": mean_motion_rev_day,
            "mean_motion_derivative": mean_motion_derivative_rev_day2,
            "orbital_speed_km_s": orbital_speed_km_s,
            "altitude_drift_km_per_day": altitude_drift_km_per_day,
            "bstar": model.bstar,
            "inclination_deg": math.degrees(model.inclo),
            "raan_deg": math.degrees(model.nodeo),
            "eccentricity": model.ecco,
            "arg_perigee_deg": math.degrees(model.argpo),
            "mean_anomaly_deg": math.degrees(model.mo),
            "rev_number": model.revnum,
        }

    for idx, (epoch_dt, name, l1, l2) in enumerate(tles):
        sat = EarthSatellite(l1, l2, name, ts)
        tle_params.append(extract_tle_params(sat))

        if idx < len(tles) - 1:
            next_epoch_dt = tles[idx + 1][0]
        else:
            next_epoch_dt = epoch_dt + timedelta(hours=extra_hours_last)

        print(f"TLE #{idx:02d} : propagation de {epoch_dt} à {next_epoch_dt}")

        current_dt = epoch_dt
        while current_dt < next_epoch_dt:
            all_times.append(current_dt)
            all_tle_idx.append(idx)

            dt_sec = (current_dt - epoch_dt).total_seconds()
            all_dt_since.append(dt_sec)

            t_sf = ts.utc(current_dt.year, current_dt.month, current_dt.day,
                          current_dt.hour, current_dt.minute, current_dt.second)
            geo = sat.at(t_sf)
            x, y, z = geo.position.km
            all_positions.append((x, y, z))

            current_dt = current_dt + timedelta(minutes=step_minutes)

    sgp4_pos = np.array(all_positions)
    dt_since_tle = np.array(all_dt_since)
    return all_times, sgp4_pos, all_tle_idx, dt_since_tle, tle_params


# =========================
# 4) Positions Horizons (robuste : chunks fixes + retries + NaN)
# =========================

def fetch_horizons_positions(times_dt, horizons_id=HORIZONS_ID):
    """
    Récupère les positions Horizons pour une liste de datetime.
    - Convertit en JD TDB (comme attendu par Horizons pour .vectors())
    - Requête en chunks fixes de HORIZONS_CHUNK_SIZE (20 par 20)
      avec retries en cas de HTTPError (ex: 502)
    - Demande un repère équatorial (refplane='earth') pour matcher SGP4.

    Retourne np.ndarray (N,3) en km (avec éventuellement des NaN).
    """
    print(
        f"Requête Horizons pour {len(times_dt)} dates "
        f"(chunks de {HORIZONS_CHUNK_SIZE}, retries)..."
    )

    # 1) Conversion des datetimes UTC -> JD en TDB (échelle de temps Horizons vectors)
    t = Time(times_dt, scale="utc")
    times_jd_tdb = t.tdb.jd

    positions = []  # même ordre que times_jd_tdb

    for start_idx in range(0, len(times_jd_tdb), HORIZONS_CHUNK_SIZE):
        jd_chunk = times_jd_tdb[start_idx:start_idx + HORIZONS_CHUNK_SIZE]
        print(
            f"-> chunk de {len(jd_chunk)} epochs "
            f"(positions {start_idx} à {start_idx + len(jd_chunk) - 1})"
        )

        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                obj = Horizons(
                    id=horizons_id,
                    location="500@399",      # géocentre terrestre
                    epochs=jd_chunk
                )
                # 2) Référentiel équatorial (compatible ICRS/GCRS)
                vec = obj.vectors(refplane="earth", delta_T=True)

                AU_KM = 149597870.7
                x = np.array(vec["x"]) * AU_KM
                y = np.array(vec["y"]) * AU_KM
                z = np.array(vec["z"]) * AU_KM

                for i in range(len(x)):
                    positions.append([x[i], y[i], z[i]])

                success = True
                break

            except HTTPError as e:
                print(f"  HTTPError tentative {attempt}/{MAX_RETRIES} : {e}")
                if attempt < MAX_RETRIES:
                    print(f"    -> pause {RETRY_SLEEP_SECONDS}s puis retry")
                    time.sleep(RETRY_SLEEP_SECONDS)
                else:
                    print(f"    -> échec après {MAX_RETRIES} tentatives")

        if not success:
            print(f"  !! Échec définitif pour ce chunk -> NaN pour {len(jd_chunk)} epochs")
            for _ in jd_chunk:
                positions.append([np.nan, np.nan, np.nan])

    positions = np.array(positions)
    if positions.shape[0] != len(times_jd_tdb):
        raise RuntimeError(
            f"Incohérence : {positions.shape[0]} positions pour {len(times_jd_tdb)} dates"
        )

    return positions


# =========================
# 5) Construction du dataset
# =========================

def build_dataset():
    # 1) TLE
    tles = load_tles_from_csv(CSV_TLE_FILE)
    print(f"{len(tles)} TLE chargés depuis {CSV_TLE_FILE}")

    # 2) SGP4
    times_dt, sgp4_pos, tle_idx, dt_since_tle, tle_params = generate_sgp4_trajectory(tles)

    # 3) Horizons
    horizons_pos = fetch_horizons_positions(times_dt)

    if sgp4_pos.shape != horizons_pos.shape:
        raise RuntimeError("Dimensions différentes entre SGP4 et Horizons")

    # 4) Erreurs
    diff = sgp4_pos - horizons_pos
    err_norm = np.linalg.norm(diff, axis=1)

    # 5) Infos TLE associées
    tle_epochs = [tles[i][0] for i in tle_idx]
    mean_motion = [tle_params[i]["mean_motion"] for i in tle_idx]
    mean_motion_deriv = [tle_params[i]["mean_motion_derivative"] for i in tle_idx]
    orbital_speed_km_s = [tle_params[i]["orbital_speed_km_s"] for i in tle_idx]
    altitude_drift_km_per_day = [tle_params[i]["altitude_drift_km_per_day"] for i in tle_idx]
    bstar = [tle_params[i]["bstar"] for i in tle_idx]
    inclination_deg = [tle_params[i]["inclination_deg"] for i in tle_idx]
    raan_deg = [tle_params[i]["raan_deg"] for i in tle_idx]
    eccentricity = [tle_params[i]["eccentricity"] for i in tle_idx]
    arg_perigee_deg = [tle_params[i]["arg_perigee_deg"] for i in tle_idx]
    mean_anomaly_deg = [tle_params[i]["mean_anomaly_deg"] for i in tle_idx]
    rev_number = [tle_params[i]["rev_number"] for i in tle_idx]

    # 6) DataFrame final
    df = pd.DataFrame({
        "time_utc": [dt.isoformat() for dt in times_dt],
        "tle_index": tle_idx,
        "tle_epoch": [e.isoformat() for e in tle_epochs],
        "dt_since_tle_s": dt_since_tle,
        "mean_motion": mean_motion,                      # rev/jour - vitesse orbitale moyenne
        "orbital_speed_km_s": orbital_speed_km_s,        # estimation de la vitesse orbitale instantanée
        "mean_motion_derivative": mean_motion_deriv,     # rev/jour^2 - dérive de période orbitale
        "altitude_drift_km_per_day": altitude_drift_km_per_day,    # perte (ou gain) d'altitude estimée
        "bstar": bstar,                                  # traînée (modèle SGP4)
        "inclination_deg": inclination_deg,              # orientation du plan orbital
        "raan_deg": raan_deg,                            # orientation du plan (noeud ascendant)
        "eccentricity": eccentricity,                    # forme de l'orbite
        "arg_perigee_deg": arg_perigee_deg,              # orientation de l'ellipse
        "mean_anomaly_deg": mean_anomaly_deg,            # position moyenne sur l'orbite
        "rev_number": rev_number,                        # numéro de révolution
        "x_sgp4_km": sgp4_pos[:, 0],
        "y_sgp4_km": sgp4_pos[:, 1],
        "z_sgp4_km": sgp4_pos[:, 2],
        "x_horizons_km": horizons_pos[:, 0],
        "y_horizons_km": horizons_pos[:, 1],
        "z_horizons_km": horizons_pos[:, 2],
        "dx_km": diff[:, 0],
        "dy_km": diff[:, 1],
        "dz_km": diff[:, 2],
        "error_norm_km": err_norm,
    })

    df.to_csv(OUTPUT_DATASET, sep=";", index=False)
    print(f"Dataset écrit dans {OUTPUT_DATASET}")
    print(df.head())


if __name__ == "__main__":
    build_dataset()
