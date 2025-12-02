"""
Phase 1: Train predictive LSTM/GRU model for ISS state forecasting (t+1).
End-to-end script: loads TERRA.csv, builds sliding-window dataset, trains model
with early stopping, evaluates MSE/RMSE, and saves the trained weights.
"""
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# -----------------------------
# Configuration dataclass
# -----------------------------
@dataclass
class TrainConfig:
    data_path: str = "TERRA.csv"
    window_size: int = 20
    batch_size: int = 64
    hidden_size: int = 128
    num_layers: int = 2
    learning_rate: float = 1e-3
    num_epochs: int = 50
    patience: int = 5
    model_type: str = "lstm"  # or "gru"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    model_path: str = "phase1_model.pth"


# -----------------------------
# Utility functions
# -----------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(path: str) -> pd.DataFrame:
    """Load CSV containing orbital states. Generates synthetic data if file missing."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        expected_cols = [
            "true_x_km",
            "true_y_km",
            "true_z_km",
            "true_vx_km_s",
            "true_vy_km_s",
            "true_vz_km_s",
            "epoch_utc",
        ]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        df = df.sort_values("epoch_utc").reset_index(drop=True)
        return df

    # Fallback synthetic circular orbit for demo
    print(f"[WARN] {path} not found. Generating synthetic circular orbit data.")
    num_points = 500
    t = np.linspace(0, 2 * np.pi, num_points)
    radius_km = 6800
    angular_rate = 2 * np.pi / (92 * 60)  # ~92 minute orbit
    df = pd.DataFrame(
        {
            "true_x_km": radius_km * np.cos(t),
            "true_y_km": radius_km * np.sin(t),
            "true_z_km": np.zeros_like(t),
            "true_vx_km_s": -radius_km * angular_rate * np.sin(t),
            "true_vy_km_s": radius_km * angular_rate * np.cos(t),
            "true_vz_km_s": np.zeros_like(t),
            "epoch_utc": pd.date_range("2024-01-01", periods=num_points, freq="T"),
        }
    )
    return df


def create_sequences(
    data: np.ndarray, window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert normalized data into sliding windows (X) and next-step targets (y)."""
    xs, ys = [], []
    for i in range(len(data) - window_size):
        x_seq = data[i : i + window_size] #L états consécutifs
        y = data[i + window_size] #état suivant
        xs.append(x_seq)
        ys.append(y)
    return np.stack(xs), np.stack(ys)


class OrbitDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.targets[idx]


# -----------------------------
# Models
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out


class GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# -----------------------------
# Training / Evaluation
# -----------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
) -> Tuple[nn.Module, List[float]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    best_val = float("inf")
    patience_counter = 0
    history = []

    model.to(cfg.device)
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                preds = model(xb)
                val_losses.append(criterion(preds, yb).item())

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))
        history.append((avg_train, avg_val))
        print(f"Epoch {epoch:03d} | train={avg_train:.6f} | val={avg_val:.6f}")

        if avg_val < best_val - 1e-6:
            best_val = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), cfg.model_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print("Early stopping triggered.")
                break

    # Load best
    model.load_state_dict(torch.load(cfg.model_path, map_location=cfg.device))
    return model, history


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    cfg: TrainConfig,
    mean: np.ndarray,
    std: np.ndarray,
    split_name: str = "Test",
):
    """
    Évalue le modèle sur un DataLoader donné (train / val / test)
    et affiche un ensemble de métriques physiques.

    Métriques :
    1) MSE normalisée (espace z-score)
    2) RMSE globale (toutes features dénormalisées, mélange km et km/s)
    3) RMSE par dimension (x, y, z, vx, vy, vz)
    4) Position RMSE 3D (km)
    5) Velocity RMSE 3D (km/s)
    6) Energy RMSE (erreur sur l'énergie mécanique spécifique)
    7) Radius RMSE (km)
    8) Angular RMSE (rad)
    9) Relative position error moyen (sans unité)
    """
    model.eval()
    criterion = nn.MSELoss()
    losses = []
    preds_list, targets_list = [], []

    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            pred = model(xb)
            losses.append(criterion(pred, yb).item())
            preds_list.append(pred.cpu().numpy())
            targets_list.append(yb.cpu().numpy())

    # 1) MSE normalisée
    mse_norm = float(np.mean(losses))

    # Concaténation des batchs
    preds = np.concatenate(preds_list, axis=0)     # (N, 6) normalisé
    targets = np.concatenate(targets_list, axis=0) # (N, 6) normalisé

    # Dénormalisation
    preds_phys = preds * std + mean
    targets_phys = targets * std + mean

    # 2) RMSE globale (toutes features, mélange km & km/s)
    rmse_global = math.sqrt(np.mean((preds_phys - targets_phys) ** 2))

    # 3) RMSE par dimension
    err_sq = (preds_phys - targets_phys) ** 2  # (N, 6)
    rmse_per_dim = np.sqrt(err_sq.mean(axis=0))  # (6,)

    # 4) Position RMSE 3D (km)
    pos_pred = preds_phys[:, :3]   # (N, 3)
    pos_true = targets_phys[:, :3]
    pos_err_sq = np.sum((pos_pred - pos_true) ** 2, axis=1)  # (N,)
    pos_rmse = float(np.sqrt(pos_err_sq.mean()))

    # 5) Velocity RMSE 3D (km/s)
    vel_pred = preds_phys[:, 3:]   # (N, 3)
    vel_true = targets_phys[:, 3:]
    vel_err_sq = np.sum((vel_pred - vel_true) ** 2, axis=1)
    vel_rmse = float(np.sqrt(vel_err_sq.mean()))

    # 6) Energy RMSE
    mu = 398600.4418  # km^3/s^2
    r_true = np.linalg.norm(pos_true, axis=1)  # (N,)
    r_pred = np.linalg.norm(pos_pred, axis=1)
    v2_true = np.sum(vel_true ** 2, axis=1)
    v2_pred = np.sum(vel_pred ** 2, axis=1)
    eps_true = v2_true / 2.0 - mu / r_true
    eps_pred = v2_pred / 2.0 - mu / r_pred
    eps_rmse = float(np.sqrt(np.mean((eps_pred - eps_true) ** 2)))

    # 7) Radius RMSE (km)
    radius_rmse = float(np.sqrt(np.mean((r_pred - r_true) ** 2)))

    # 8) Angular RMSE (rad) dans le plan (x, y)
    theta_true = np.arctan2(pos_true[:, 1], pos_true[:, 0])
    theta_pred = np.arctan2(pos_pred[:, 1], pos_pred[:, 0])
    dtheta = np.arctan2(np.sin(theta_pred - theta_true), np.cos(theta_pred - theta_true))
    angular_rmse = float(np.sqrt(np.mean(dtheta ** 2)))

    # 9) Relative position error moyen
    rel_pos_err = np.linalg.norm(pos_pred - pos_true, axis=1) / np.linalg.norm(pos_true, axis=1)
    rel_pos_err_mean = float(rel_pos_err.mean())

    feature_cols = [
        "true_x_km",
        "true_y_km",
        "true_z_km",
        "true_vx_km_s",
        "true_vy_km_s",
        "true_vz_km_s",
    ]

    print(f"\n===== {split_name} METRICS =====")
    print(f"MSE (normalized)                : {mse_norm:.6f}")
    print(f"Global RMSE (all features phys) : {rmse_global:.6f}")
    print("\nRMSE per dimension:")
    for name, val in zip(feature_cols, rmse_per_dim):
        print(f"  {name:15s}: {val:.6f}")
    print(f"\nPosition RMSE 3D (km)           : {pos_rmse:.6f}")
    print(f"Velocity RMSE 3D (km/s)         : {vel_rmse:.6f}")
    print(f"Energy RMSE (km^2/s^2)          : {eps_rmse:.6f}")
    print(f"Radius RMSE (km)                : {radius_rmse:.6f}")
    print(f"Angular RMSE (rad)              : {angular_rmse:.6f}")
    print(f"Mean relative position error    : {rel_pos_err_mean:.6f}")
    print("================================\n")

    metrics = {
        "mse_norm": mse_norm,
        "rmse_global": rmse_global,
        "rmse_per_dim": rmse_per_dim.tolist(),
        "pos_rmse_km": pos_rmse,
        "vel_rmse_km_s": vel_rmse,
        "energy_rmse": eps_rmse,
        "radius_rmse_km": radius_rmse,
        "angular_rmse_rad": angular_rmse,
        "rel_pos_err_mean": rel_pos_err_mean,
    }
    return metrics


# -----------------------------
# Main execution
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train LSTM/GRU orbital predictor (t+1)")
    parser.add_argument("--data", default="/Users/bilaldelais/Desktop/project deep learning/data/TERRA.csv", help="Path to dataset CSV")
    parser.add_argument("--model", choices=["lstm", "gru"], help="(deprecated) Train a single model type")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lstm", "gru"],
        default=None,
        help="List of models to train/evaluate (default: both)",
    )
    parser.add_argument("--window", type=int, default=20, help="Sliding window length L")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size")
    parser.add_argument("--layers", type=int, default=2, help="Number of recurrent layers")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save", default="phase1_model.pth", help="Output model path")
    args = parser.parse_args()

    models_to_run = args.models or ([args.model] if args.model else ["lstm", "gru"])

    base_cfg = TrainConfig(
        data_path=args.data,
        window_size=args.window,
        batch_size=args.batch,
        hidden_size=args.hidden,
        num_layers=args.layers,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        patience=args.patience,
        model_path=args.save,
    )

    set_seed(base_cfg.seed)

    df = load_data(base_cfg.data_path)
    feature_cols = [
        "true_x_km",
        "true_y_km",
        "true_z_km",
        "true_vx_km_s",
        "true_vy_km_s",
        "true_vz_km_s",
    ]
    data = df[feature_cols].to_numpy().astype(np.float32)

    # Train/Val/Test split (chronological): 70/15/15
    n = len(data)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train_data = data[:n_train]
    val_data = data[n_train : n_train + n_val]
    test_data = data[n_train + n_val :]

    # Normalization (fit on train only)
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0) + 1e-8
    train_norm = (train_data - mean) / std
    val_norm = (val_data - mean) / std
    test_norm = (test_data - mean) / std

    # Sliding windows
    x_train, y_train = create_sequences(train_norm, base_cfg.window_size)
    x_val, y_val = create_sequences(val_norm, base_cfg.window_size)
    x_test, y_test = create_sequences(test_norm, base_cfg.window_size)

    # DataLoaders
    train_loader = DataLoader(OrbitDataset(x_train, y_train), batch_size=base_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(OrbitDataset(x_val, y_val), batch_size=base_cfg.batch_size, shuffle=False)
    test_loader = DataLoader(OrbitDataset(x_test, y_test), batch_size=base_cfg.batch_size, shuffle=False)

    input_size = len(feature_cols)
    output_size = len(feature_cols)
    save_root, save_ext = os.path.splitext(args.save)
    default_ext = save_ext if save_ext else ".pth"
    results = {}

    for model_type in models_to_run:
        model_path = args.save if len(models_to_run) == 1 else f"{save_root}_{model_type}{default_ext}"
        cfg = TrainConfig(
            data_path=base_cfg.data_path,
            window_size=base_cfg.window_size,
            batch_size=base_cfg.batch_size,
            hidden_size=base_cfg.hidden_size,
            num_layers=base_cfg.num_layers,
            learning_rate=base_cfg.learning_rate,
            num_epochs=base_cfg.num_epochs,
            patience=base_cfg.patience,
            model_type=model_type,
            device=base_cfg.device,
            seed=base_cfg.seed,
            model_path=model_path,
        )

        # Reseed to keep runs comparable across models
        set_seed(cfg.seed)

        if model_type == "lstm":
            model = LSTMModel(input_size, cfg.hidden_size, cfg.num_layers, output_size)
        else:
            model = GRUModel(input_size, cfg.hidden_size, cfg.num_layers, output_size)

        print(f"\n=== Training {model_type.upper()} model ===")
        print(model)
        model, history = train_model(model, train_loader, val_loader, cfg)

        # Évaluation complète par modèle : Train / Val / Test
        train_metrics = evaluate_model(
            model,
            train_loader,
            cfg,
            mean,
            std,
            split_name=f"{model_type.upper()} / Train",
        )
        val_metrics = evaluate_model(
            model,
            val_loader,
            cfg,
            mean,
            std,
            split_name=f"{model_type.upper()} / Val",
        )
        test_metrics = evaluate_model(
            model,
            test_loader,
            cfg,
            mean,
            std,
            split_name=f"{model_type.upper()} / Test",
        )

        # Save model + normalization metadata for downstream phases
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_type": cfg.model_type,
                "input_size": input_size,
                "hidden_size": cfg.hidden_size,
                "num_layers": cfg.num_layers,
                "window_size": cfg.window_size,
                "mean": mean,
                "std": std,
                "feature_cols": feature_cols,
            },
            cfg.model_path,
        )
        print(f"[{model_type.upper()}] Model and metadata saved to {cfg.model_path}")
        results[model_type] = {
            "history": history,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "model_path": cfg.model_path,
        }

    print("\n=== Comparison (test set) ===")
    for model_type in models_to_run:
        res = results[model_type]
        tm = res["test_metrics"]
        print(
            f"{model_type.upper()}: "
            f"MSE_norm={tm['mse_norm']:.6f} | "
            f"Pos_RMSE_km={tm['pos_rmse_km']:.3f} | "
            f"Vel_RMSE_km_s={tm['vel_rmse_km_s']:.6f} | "
            f"saved={res['model_path']}"
        )

    with open("training_history.json", "w") as f:
        json.dump({"models": results}, f, indent=2)


if __name__ == "__main__":
    main()
