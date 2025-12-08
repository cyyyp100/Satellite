import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List
import pandas as pd
import time 
import os

os.makedirs("results", exist_ok=True)

# =========================================================
#  DATASET 70 / 15 / 15
# =========================================================
class SatelliteSequenceDataset:
    def __init__(self, X, Y, seq_len=128):
        self.X = X
        self.Y = Y
        self.seq_len = seq_len

        n = len(X)
        self.n_train = int(0.70 * n)
        self.n_valid = int(0.15 * n)
        self.n_test  = n - self.n_train - self.n_valid

        self.X_train = X[: self.n_train]
        self.Y_train = Y[: self.n_train]

        self.X_valid = X[self.n_train : self.n_train + self.n_valid]
        self.Y_valid = Y[self.n_train : self.n_train + self.n_valid]

        self.X_test  = X[self.n_train + self.n_valid :]
        self.Y_test  = Y[self.n_train + self.n_valid :]

        print(f"Dataset split: train={len(self.X_train)}, valid={len(self.X_valid)}, test={len(self.X_test)}")

    def get_loader(self, subset, batch_size=32, shuffle=False):
        X = getattr(self, f"X_{subset}")
        Y = getattr(self, f"Y_{subset}")

        class SubsetDataset(Dataset):
            def __len__(self2):
                return len(X) - self.seq_len

            def __getitem__(self2, idx):
                X_seq = X[idx : idx + self.seq_len]
                y = Y[idx + self.seq_len - 1]
                return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

        return DataLoader(SubsetDataset(), batch_size=batch_size, shuffle=shuffle)


# =========================================================
#  MODEL LSTM
# =========================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# =========================================================
#  TRAINER (ajout test set)
# =========================================================
class LSTMTrainer:
    def __init__(self, model, lr=1e-3, device="mps"):
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        print("Using:", self.device)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.history = {
            "train_mse": [], "train_rmse": [],
            "valid_mse": [], "valid_rmse": [],
            "test_mse": [],  "test_rmse": [],
            "epoch_time": []
        }

    def eval_epoch(self, loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                loss = self.loss_fn(self.model(X), y)
                losses.append(loss.item())
        mse = np.mean(losses)
        return mse, np.sqrt(mse)

    def train_epoch(self, loader):
        self.model.train()
        losses = []
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        mse = np.mean(losses)
        return mse, np.sqrt(mse)

    def fit(self, train_loader, valid_loader, test_loader, epochs=150):
        for epoch in range(epochs):
            t0 = time.time()

            train_mse, train_rmse = self.train_epoch(train_loader)
            valid_mse, valid_rmse = self.eval_epoch(valid_loader)
            test_mse,  test_rmse  = self.eval_epoch(test_loader)

            dt = time.time() - t0

            self.history["train_mse"].append(train_mse)
            self.history["train_rmse"].append(train_rmse)

            self.history["valid_mse"].append(valid_mse)
            self.history["valid_rmse"].append(valid_rmse)

            self.history["test_mse"].append(test_mse)
            self.history["test_rmse"].append(test_rmse)

            self.history["epoch_time"].append(dt)

            print(
                f"[EPOCH {epoch+1:03d}] "
                f"train={train_mse:.5f} | "
                f"valid={valid_mse:.5f} | "
                f"test={test_mse:.5f} | "
                f"time={dt:.2f}s"
            )

        return self.history


# =========================================================
#  CODE PRINCIPAL
# =========================================================
if __name__ == "__main__":

    df = pd.read_csv("datasetISS_200TLE.csv", sep=";")

    for c in ["dx_km", "dy_km", "dz_km"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    X_cols = [
        "time_utc", "tle_index", "tle_epoch", "dt_since_tle_s",
        "mean_motion", "orbital_speed_km_s", "mean_motion_derivative",
        "altitude_drift_km_per_day", "bstar", "inclination_deg",
        "raan_deg", "eccentricity", "arg_perigee_deg",
        "mean_anomaly_deg", "rev_number",
        "x_sgp4_km", "y_sgp4_km", "z_sgp4_km"
    ]

    df["time_utc"] = pd.to_datetime(df["time_utc"]).astype("int64") / 1e9
    df["tle_epoch"] = pd.to_datetime(df["tle_epoch"]).astype("int64") / 1e9

    # Find Horizons columns
    def find_col(names):
        cols = [c for c in df.columns]
        for n in names:
            for c in cols:
                if n.lower() in c.lower():
                    return c
        raise Exception("Missing horizons column", names)

    x_h = find_col(["x_horizons"])
    y_h = find_col(["y_horizons"])
    z_h = find_col(["z_horizons"])

    df["err_x"] = df[x_h] - df["x_sgp4_km"]
    df["err_y"] = df[y_h] - df["y_sgp4_km"]
    df["err_z"] = df[z_h] - df["z_sgp4_km"]

    X = df[X_cols].values.astype(np.float32)
    Y = df[["err_x", "err_y", "err_z"]].values.astype(np.float32)

    # Normalisation
    X_mean = X.mean(0, keepdims=True)
    X_std  = X.std(0, keepdims=True) + 1e-8
    X = (X - X_mean) / X_std

    seq_len = 128
    batch_size = 32

    dataset = SatelliteSequenceDataset(X, Y, seq_len=seq_len)

    train_loader = dataset.get_loader("train", batch_size=batch_size, shuffle=True)
    valid_loader = dataset.get_loader("valid", batch_size=batch_size, shuffle=False)
    test_loader  = dataset.get_loader("test",  batch_size=batch_size, shuffle=False)

    # =========================================================
    #  EXPERIMENTS
    # =========================================================
    experiments = [
        {"hidden": 64,  "layers": 1, "lr": 1e-3},
        {"hidden": 128, "layers": 3, "lr": 1e-4},
    ]

    for cfg in experiments:
        model = LSTMModel(
            input_size=X.shape[1],
            hidden_size=cfg["hidden"],
            num_layers=cfg["layers"]
        )

        trainer = LSTMTrainer(model, lr=cfg["lr"])
        hist = trainer.fit(train_loader, valid_loader, test_loader, epochs=80)

        # CSV METRICS
        df_metrics = pd.DataFrame({
            "Train_MSE": hist["train_mse"],
            "Train_RMSE": hist["train_rmse"],
            "Val_MSE": hist["valid_mse"],
            "Val_RMSE": hist["valid_rmse"],
            "Test_MSE": hist["test_mse"],
            "Test_RMSE": hist["test_rmse"],
            "Time": hist["epoch_time"]
        })

        df_metrics.index = np.arange(1, len(df_metrics)+1)
        df_metrics.index.name = "Epoch"

        out_csv = f"results/Metrics_LSTM_H{cfg['hidden']}_L{cfg['layers']}.csv"
        df_metrics.to_csv(out_csv)
        print("Saved metrics:", out_csv)

