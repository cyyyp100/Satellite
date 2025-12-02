import argparse
import os
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from GRU import GRUModel
from LSTM import LSTMModel
from MLP import MLPModel


# ============================================================
# Dataset simple, qui reçoit des séquences déjà construites
# ============================================================

class SequenceDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, Y_seq: np.ndarray):
        self.X_seq = X_seq
        self.Y_seq = Y_seq

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_seq[idx], dtype=torch.float32),
            torch.tensor(self.Y_seq[idx], dtype=torch.float32),
        )


# ============================================================
# Fonctions utilitaires
# ============================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparer GRU, LSTM et MLP")
    parser.add_argument("--csv-path", default="datasetISS_200TLE.csv")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--mlp-hidden", default="256,256,128")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
    return parser.parse_args()


def _find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        key = c.lower()
        if key in cols_lower:
            return cols_lower[key]
    for key, original in cols_lower.items():
        for c in candidates:
            if c.lower() in key:
                return original
    raise KeyError(f"Colonne non trouvée dans {candidates}")


# ============================================================
# Chargement et préparation des données
# ============================================================

def load_and_prepare_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, sep=";")

    # Supprimer colonnes inutiles
    for col in ["dx_km", "dy_km", "dz_km"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Convertir les dates
    if "time_utc" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time_utc"]).astype("int64") / 1e9
    if "tle_epoch" in df.columns:
        df["tle_epoch"] = pd.to_datetime(df["tle_epoch"]).astype("int64") / 1e9

    # Colonnes d'entrée
    feature_cols = [
        "time_utc",
        "tle_epoch",
        "dt_since_tle_s",
        "mean_motion",
        "orbital_speed_km_s",
        "mean_motion_derivative",
        "altitude_drift_km_per_day",
        "bstar",
        "inclination_deg",
        "raan_deg",
        "eccentricity",
        "arg_perigee_deg",
        "mean_anomaly_deg",
        "rev_number",
        "x_sgp4_km",
        "y_sgp4_km",
        "z_sgp4_km",
    ]

    x_h = _find_col(df, ["x_horizons_km", "x_horizon_km", "x_horizons"])
    y_h = _find_col(df, ["y_horizons_km", "y_horizon_km", "y_horizons"])
    z_h = _find_col(df, ["z_horizons_km", "z_horizon_km", "z_horizons"])

    # Erreurs SGP4 → Horizons
    df["err_x"] = df[x_h] - df["x_sgp4_km"]
    df["err_y"] = df[y_h] - df["y_sgp4_km"]
    df["err_z"] = df[z_h] - df["z_sgp4_km"]

    X = df[feature_cols].values.astype(np.float32)
    Y = df[["err_x", "err_y", "err_z"]].values.astype(np.float32)

    # Normalisation (comme ton 1er script)
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - X_mean) / X_std

    return X, Y


# ============================================================
# Séquençage CORRECT (après split)
# ============================================================

def make_sequences(X: np.ndarray, Y: np.ndarray, seq_len: int):
    X_seq, Y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        Y_seq.append(Y[i+seq_len-1])
    return np.array(X_seq), np.array(Y_seq)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    # -----------------------------
    # 1) Charger dataset
    # -----------------------------
    X, Y = load_and_prepare_data(args.csv_path)
    N = len(X)

    train_end = int(N * args.train_ratio)
    val_end = int(N * (args.train_ratio + args.val_ratio))

    X_train, Y_train = X[:train_end], Y[:train_end]
    X_valid, Y_valid = X[train_end:val_end], Y[train_end:val_end]

    # -----------------------------
    # 2) Construire séquences COHÉRENTES
    # -----------------------------
    Xseq_train, Yseq_train = make_sequences(X_train, Y_train, args.seq_len)
    Xseq_valid, Yseq_valid = make_sequences(X_valid, Y_valid, args.seq_len)

    train_loader = DataLoader(SequenceDataset(Xseq_train, Yseq_train),
                              batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(SequenceDataset(Xseq_valid, Yseq_valid),
                            batch_size=args.batch_size, shuffle=False)

    input_size = X.shape[1]
    flattened_dim = args.seq_len * input_size

    histories = {}

    # ============================================================
    # GRU
    # ============================================================
    gru = GRUModel(
        input_size=input_size,
        hidden_size=args.gru_hidden,
        num_layers=1,
        dropout=args.dropout,
        lr=args.learning_rate,
    )
    label = f"GRU hidden={args.gru_hidden}"
    histories[label] = gru.train_model(
        train_loader, val_loader,
        epochs=args.epochs, lr=args.learning_rate,
        patience=args.patience, min_delta=args.min_delta)

    # ============================================================
    # LSTM
    # ============================================================
    lstm = LSTMModel(
        input_size=input_size,
        hidden_size=args.lstm_hidden,
        num_layers=1,
        dropout=args.dropout,
        lr=args.learning_rate,
    )
    label = f"LSTM hidden={args.lstm_hidden}"
    histories[label] = lstm.train_model(
        train_loader, val_loader,
        epochs=args.epochs, lr=args.learning_rate,
        patience=args.patience, min_delta=args.min_delta)

    # ============================================================
    # MLP
    # ============================================================
    mlp_hidden = [int(h) for h in args.mlp_hidden.split(",")]

    mlp = MLPModel(
        input_dim=flattened_dim,
        hidden_dims=mlp_hidden,
        dropout=args.dropout,
        lr=args.learning_rate,
    )
    label = f"MLP hidden={mlp_hidden}"
    histories[label] = mlp.train_model(
        train_loader, val_loader,
        epochs=args.epochs, lr=args.learning_rate,
        patience=args.patience, min_delta=args.min_delta)

    # -----------------------------
    # 3) Plots
    # -----------------------------
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(12, 5))
    for name, hist in histories.items():
        plt.plot(hist["val_rmse"], label=name)
    plt.legend()
    plt.title("Validation RMSE")
    plt.grid()
    plt.savefig("plots/val_rmse.png")

    plt.figure(figsize=(12, 5))
    for name, hist in histories.items():
        plt.plot(hist["val_loss"], label=name)
    plt.legend()
    plt.title("Validation MSE")
    plt.grid()
    plt.savefig("plots/val_mse.png")


if __name__ == "__main__":
    main()
