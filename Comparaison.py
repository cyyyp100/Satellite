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


class SatelliteSequenceDataset(Dataset):
    def __init__(
        self,
        X_seq: np.ndarray,
        Y_seq: np.ndarray,
        train_ratio: float,
        val_ratio: float,
        mode: str = "train",
        splits: Dict[str, Tuple[int, int]] | None = None,
    ):
        self.X_seq = X_seq
        self.Y_seq = Y_seq
        n_samples = X_seq.shape[0]
        if splits is None:
            train_end = int(n_samples * train_ratio)
            val_end = int(n_samples * (train_ratio + val_ratio))
            splits = {"train": (0, train_end), "valid": (train_end, val_end), "test": (val_end, n_samples)}
        self.splits = splits
        self.mode = "train"
        self.set_mode(mode)

    def set_mode(self, mode: str) -> None:
        if mode not in self.splits:
            raise ValueError(f"Mode {mode} inconnu. Utilisez 'train', 'valid' ou 'test'.")
        start, end = self.splits[mode]
        self.current_X = self.X_seq[start:end]
        self.current_Y = self.Y_seq[start:end]
        self.mode = mode

    def __len__(self) -> int:
        return self.current_X.shape[0]

    def __getitem__(self, idx: int):
        return torch.tensor(self.current_X[idx], dtype=torch.float32), torch.tensor(self.current_Y[idx], dtype=torch.float32)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparer GRU, LSTM et MLP sur le même séquençage")
    parser.add_argument("--csv-path", default="datasetISS_200TLE.csv", help="Chemin vers le CSV d'entrée")
    parser.add_argument("--seq-len", type=int, default=128, help="Longueur de séquence")
    parser.add_argument("--batch-size", type=int, default=64, help="Taille de batch partagée")
    parser.add_argument("--epochs", type=int, default=50, help="Nombre d'époques pour chaque modèle")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Proportion d'apprentissage")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Proportion de validation")
    parser.add_argument("--gru-hidden", type=int, default=64, help="Taille cachée GRU")
    parser.add_argument("--lstm-hidden", type=int, default=64, help="Taille cachée LSTM")
    parser.add_argument("--mlp-hidden", default="256,256,128", help="Dimensions cachées du MLP, séparées par des virgules")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate (utilisé par défaut pour tous les modèles)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout des modèles séquentiels")
    parser.add_argument("--seed", type=int, default=42, help="Seed globale pour la reproductibilité")
    parser.add_argument("--patience", type=int, default=10, help="Patience pour l'early stopping")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Amélioration minimale du MSE pour réinitialiser l'early stopping")
    return parser.parse_args()


def _find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in cols_lower:
            return cols_lower[key]
    for key, original in cols_lower.items():
        for cand in candidates:
            if cand.lower().strip() in key:
                return original
    raise KeyError(f"Aucune colonne trouvée parmi {candidates} dans {df.columns}")


def load_and_prepare_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, sep=";")

    for col in ["dx_km", "dy_km", "dz_km"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if "time_utc" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time_utc"]).astype("int64") / 1e9
    if "tle_epoch" in df.columns:
        df["tle_epoch"] = pd.to_datetime(df["tle_epoch"]).astype("int64") / 1e9

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

    x_h_col = _find_col(df, ["x_horizons_km", "x_horizon_km", "x_horizons"])
    y_h_col = _find_col(df, ["y_horizons_km", "y_horizon_km", "y_horizons"])
    z_h_col = _find_col(df, ["z_horizons_km", "z_horizon_km", "z_horizons"])

    df["err_x"] = df[x_h_col] - df["x_sgp4_km"]
    df["err_y"] = df[y_h_col] - df["y_sgp4_km"]
    df["err_z"] = df[z_h_col] - df["z_sgp4_km"]

    X = df[feature_cols].values.astype(np.float32)
    Y = df[["err_x", "err_y", "err_z"]].values.astype(np.float32)

    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - X_mean) / X_std

    return X, Y


def build_sequences(X: np.ndarray, Y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    samples = X.shape[0]
    X_seq: List[np.ndarray] = []
    Y_seq: List[np.ndarray] = []
    for start in range(samples - seq_len + 1):
        end = start + seq_len
        X_seq.append(X[start:end])
        Y_seq.append(Y[end - 1])
    return np.stack(X_seq), np.stack(Y_seq)


def split_sequences(
    X_seq: np.ndarray,
    Y_seq: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    n = X_seq.shape[0]
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (
        (X_seq[:train_end], Y_seq[:train_end]),
        (X_seq[train_end:val_end], Y_seq[train_end:val_end]),
        (X_seq[val_end:], Y_seq[val_end:]),
    )


def create_dataloaders(
    X_seq: np.ndarray,
    Y_seq: np.ndarray,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    template_dataset = SatelliteSequenceDataset(X_seq, Y_seq, train_ratio, val_ratio, mode="train")
    splits = template_dataset.splits

    train_dataset = SatelliteSequenceDataset(X_seq, Y_seq, train_ratio, val_ratio, mode="train", splits=splits)
    train_dataset.set_mode("train")
    val_dataset = SatelliteSequenceDataset(X_seq, Y_seq, train_ratio, val_ratio, mode="valid", splits=splits)
    val_dataset.set_mode("valid")
    test_dataset = SatelliteSequenceDataset(X_seq, Y_seq, train_ratio, val_ratio, mode="test", splits=splits)
    test_dataset.set_mode("test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def parse_mlp_hidden(hidden_str: str) -> List[int]:
    return [int(h.strip()) for h in hidden_str.split(",") if h.strip()]


def plot_histories(histories: Dict[str, Dict[str, List[float]]]) -> None:
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(10, 5))
    for name, hist in histories.items():
        plt.plot(hist["val_rmse"], label=name)
    plt.title("RMSE Validation par époque")
    plt.xlabel("Époque")
    plt.ylabel("RMSE (validation)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/validation_rmse.png")

    plt.figure(figsize=(10, 5))
    for name, hist in histories.items():
        plt.plot(hist["val_loss"], label=name)
    plt.title("MSE Validation par époque")
    plt.xlabel("Époque")
    plt.ylabel("MSE (validation)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/validation_mse.png")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("La somme train_ratio + val_ratio doit être < 1")

    X, Y = load_and_prepare_data(args.csv_path)
    X_seq, Y_seq = build_sequences(X, Y, args.seq_len)
    train_loader, val_loader, _ = create_dataloaders(X_seq, Y_seq, args.batch_size, args.train_ratio, args.val_ratio)

    input_size = X_seq.shape[2]
    flattened_dim = args.seq_len * input_size
    histories: Dict[str, Dict[str, List[float]]] = {}

    # GRU
    gru_model = GRUModel(
        input_size=input_size,
        hidden_size=args.gru_hidden,
        num_layers=1,
        dropout=args.dropout,
        lr=args.learning_rate,
    )
    gru_label = f"GRU ({args.epochs} epochs, {args.gru_hidden} blocs, learning rate = {args.learning_rate})"
    start = time.time()
    histories[gru_label] = gru_model.train_model(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.learning_rate,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    print(f"Temps total GRU : {time.time() - start:.2f}s")

    # LSTM
    lstm_model = LSTMModel(
        input_size=input_size,
        hidden_size=args.lstm_hidden,
        num_layers=1,
        dropout=args.dropout,
        lr=args.learning_rate,
    )
    lstm_label = f"LSTM ({args.epochs} epochs, {args.lstm_hidden} blocs, learning rate = {args.learning_rate})"
    start = time.time()
    histories[lstm_label] = lstm_model.train_model(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.learning_rate,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    print(f"Temps total LSTM : {time.time() - start:.2f}s")

    # MLP
    mlp_hidden = parse_mlp_hidden(args.mlp_hidden)
    mlp_model = MLPModel(
        input_dim=flattened_dim,
        hidden_dims=mlp_hidden,
        dropout=args.dropout,
        lr=args.learning_rate,
    )
    mlp_label = f"MLP ({args.epochs} epochs, hidden={mlp_hidden}, learning rate = {args.learning_rate})"
    start = time.time()
    histories[mlp_label] = mlp_model.train_model(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.learning_rate,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    print(f"Temps total MLP : {time.time() - start:.2f}s")

    plot_histories(histories)


if __name__ == "__main__":
    main()
