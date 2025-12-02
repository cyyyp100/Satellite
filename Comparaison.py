import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from GRU import GRUModel
from LSTM import LSTMModel
from MLP import MLPCorrectionModel


class GRUDataset(Dataset):
    def __init__(self, X_seq, Y_seq):
        self.X_seq = X_seq
        self.Y_seq = Y_seq

    def __len__(self):
        return self.X_seq.shape[0]

    def __getitem__(self, idx):
        return self.X_seq[idx], self.Y_seq[idx]

def build_sequences(X, Y, seq_len):
    n_samples = X.shape[0]
    X_seq = []
    Y_seq = []
    t_idx = []
    for i in range(n_samples - seq_len + 1):
        X_seq.append(X[i:i+seq_len])
        Y_seq.append(Y[i+seq_len-1])
        t_idx.append(i+seq_len-1)
    X_seq = np.stack(X_seq)
    Y_seq = np.stack(Y_seq)
    return X_seq, Y_seq, np.array(t_idx)

def train_sequence_model(name, model, train_loader, valid_loader, device, num_epochs=50):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    history = []

    for epoch in range(1, num_epochs + 1):
        start_time = time.perf_counter()
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        model.eval()
        valid_losses = []
        valid_preds = []
        valid_targets = []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                valid_losses.append(loss.item())
                valid_preds.append(preds.cpu().numpy())
                valid_targets.append(yb.cpu().numpy())
        valid_loss = float(np.mean(valid_losses))
        valid_preds = np.concatenate(valid_preds, axis=0)
        valid_targets = np.concatenate(valid_targets, axis=0)
        valid_rmse = float(np.sqrt(np.mean((valid_preds - valid_targets) ** 2)))

        epoch_time = time.perf_counter() - start_time
        print(f"[{name}][Epoch {epoch}] Train Loss={train_loss:.6f} | Valid Loss={valid_loss:.6f} | RMSE={valid_rmse:.6f} | Time={epoch_time:.2f}s")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_rmse": valid_rmse,
                "time": epoch_time,
            }
        )

    return history

def main():
    # 1. Charger CSV, construire X, Y, sgp4_pos, horizons_pos
    df = pd.read_csv("datasetISS_200TLE.csv", sep=";")

    # Supprimer dx, dy, dz si présents
    for col in ["dx_km", "dy_km", "dz_km"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Conversion des dates en timestamps numériques (secondes)
    if "time_utc" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time_utc"]).astype("int64") / 1e9
    if "tle_epoch" in df.columns:
        df["tle_epoch"] = pd.to_datetime(df["tle_epoch"]).astype("int64") / 1e9

    # Colonnes d'entrée X : mêmes que dans GRU.py / xgboost.py
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

    # Calcul des erreurs SGP4 -> Horizons
    df["err_x"] = df["x_horizons_km"] - df["x_sgp4_km"]
    df["err_y"] = df["y_horizons_km"] - df["y_sgp4_km"]
    df["err_z"] = df["z_horizons_km"] - df["z_sgp4_km"]

    # Matrices numpy pour les modèles
    X = df[feature_cols].values.astype(np.float32)
    Y = df[["err_x", "err_y", "err_z"]].values.astype(np.float32)

    # Positions SGP4 et Horizons (vérités terrain)
    sgp4_pos = df[["x_sgp4_km", "y_sgp4_km", "z_sgp4_km"]].values.astype(np.float32)
    horizons_pos = df[["x_horizons_km", "y_horizons_km", "z_horizons_km"]].values.astype(np.float32)

    # Normalisation simple (z-score) de X
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - X_mean) / X_std

    n = X.shape[0]
    # 2. Split train/valid/test avec indices globaux (70/15/15)
    train_end = int(n*0.7)
    valid_end = int(n*0.85)

    # 3. Construire séquences GRU + DataLoaders
    seq_len = 128
    X_seq, Y_seq, t_idx = build_sequences(X, Y, seq_len)

    # Déterminer les indices pour train/valid/test selon t_idx (cible temporelle)
    train_mask = (t_idx < train_end)
    valid_mask = (t_idx >= train_end) & (t_idx < valid_end)
    test_mask = (t_idx >= valid_end)

    X_train = X_seq[train_mask]
    Y_train = Y_seq[train_mask]
    X_valid = X_seq[valid_mask]
    Y_valid = Y_seq[valid_mask]
    X_test = X_seq[test_mask]
    Y_test = Y_seq[test_mask]
    t_idx_test = t_idx[test_mask]

    batch_size = 64
    train_dataset = GRUDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    valid_dataset = GRUDataset(torch.from_numpy(X_valid), torch.from_numpy(Y_valid))
    test_dataset = GRUDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Device : utiliser MPS sur Mac si disponible, sinon CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 4. Entraîner GRU, LSTM et MLP (modèles importés depuis GRU.py, LSTM.py, MLP.py)
    num_epochs = 50  # plus léger pour un test comparatif

    gru_model = GRUCorrectionModel(input_dim=len(feature_cols)).to(device)
    lstm_model = LSTMCorrectionModel(input_dim=len(feature_cols)).to(device)
    mlp_model = MLPCorrectionModel(input_dim=len(feature_cols)).to(device)

    histories = {}
    histories["GRU"] = train_sequence_model("GRU", gru_model, train_loader, valid_loader, device, num_epochs=num_epochs)
    histories["LSTM"] = train_sequence_model("LSTM", lstm_model, train_loader, valid_loader, device, num_epochs=num_epochs)
    histories["MLP"] = train_sequence_model("MLP", mlp_model, train_loader, valid_loader, device, num_epochs=num_epochs)

    # 5. Tracer les graphiques comparatifs (RMSE valid, train loss)
    fig, ax = plt.subplots()
    for name, hist in histories.items():
        ax.plot([h["epoch"] for h in hist], [h["valid_rmse"] for h in hist], label=f"{name} Valid RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE")
    ax.legend()
    ax.grid(True)
    plt.title("Validation RMSE vs Epoch pour GRU / LSTM / MLP")
    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots()
    for name, hist in histories.items():
        ax2.plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist], label=f"{name} Train Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Train Loss (MSE)")
    ax2.legend()
    ax2.grid(True)
    plt.title("Train Loss vs Epoch pour GRU / LSTM / MLP")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
