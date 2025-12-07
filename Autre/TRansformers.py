import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time
import os

os.makedirs("results", exist_ok=True)

# =========================================================
#  POSITIONAL ENCODING
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)

        self.pe = pe.unsqueeze(0)  # shape : (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


# =========================================================
#  DATASET 70 / 15 / 15
# =========================================================
class SatelliteSequenceDataset:
    def __init__(self, X, Y, seq_len=50):
        self.X = X
        self.Y = Y
        self.seq_len = seq_len

        n = len(X)
        self.n_train = int(0.60 * n)
        self.n_valid = int(0.20 * n)
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
#  TRANSFORMER ORBITAL MODEL
# =========================================================
class TransformerOrbital(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=3, dim_feedforward=256):
        super().__init__()

        self.embed = nn.Linear(input_size, d_model)
        self.posenc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 3)

    def forward(self, x):
        # x : (batch, seq_len, input_size)
        x = self.embed(x)
        x = self.posenc(x)
        out = self.encoder(x)
        last = out[:, -1, :]  # uniquement le dernier pas
        return self.fc(last)


# =========================================================
#  TRAINER
# =========================================================
class TransformerTrainer:
    def __init__(self, model, lr=1e-4, device="mps"):
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        print("Using device:", self.device)

        self.model = model.to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        self.history = {
            "train_mse": [], "train_rmse": [],
            "valid_mse": [], "valid_rmse": [],
            "test_mse":  [], "test_rmse": [],
            "epoch_time": []
        }

    def eval_epoch(self, loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
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

    def fit(self, train_loader, valid_loader, test_loader, epochs=100):
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
#  MAIN CODE
# =========================================================
if __name__ == "__main__":

    df = pd.read_csv("dataset_hst_sgp4_vs_horizons2.csv", sep=";")

    # Retire dx/dy/dz si existants
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

    def find_col(names):
        for n in names:
            for c in df.columns:
                if n.lower() in c.lower():
                    return c
        raise Exception("Missing Horizons column")

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

    seq_len = 50
    batch_size = 32

    dataset = SatelliteSequenceDataset(X, Y, seq_len=seq_len)

    train_loader = dataset.get_loader("train", batch_size=batch_size, shuffle=True)
    valid_loader = dataset.get_loader("valid", batch_size=batch_size, shuffle=False)
    test_loader  = dataset.get_loader("test",  batch_size=batch_size, shuffle=False)

    model = TransformerOrbital(
        input_size=X.shape[1],
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256
    )

    trainer = TransformerTrainer(model, lr=1e-4)
    hist = trainer.fit(train_loader, valid_loader, test_loader, epochs=120)

    # Save METRICS CSV
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

    metrics_path = "results/Metrics_Transformer.csv"
    df_metrics.to_csv(metrics_path)
    print("Saved metrics:", metrics_path)
