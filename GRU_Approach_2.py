import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time
import os

os.makedirs("results", exist_ok=True)

# =========================================================
#  DATASET SEQUENTIEL
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

        self.X_train = X[:self.n_train]
        self.Y_train = Y[:self.n_train]

        self.X_valid = X[self.n_train : self.n_train + self.n_valid]
        self.Y_valid = Y[self.n_train : self.n_train + self.n_valid]

        self.X_test  = X[self.n_train + self.n_valid :]
        self.Y_test  = Y[self.n_train + self.n_valid :]

        print(f"[Dataset] Train={len(self.X_train)}, Valid={len(self.X_valid)}, Test={len(self.X_test)}")

    def get_loader(self, subset, batch_size=32, shuffle=False):
        X = getattr(self, f"X_{subset}")
        Y = getattr(self, f"Y_{subset}")

        class SeqDataset(Dataset):
            def __len__(self2):
                return len(X) - self.seq_len

            def __getitem__(self2, idx):
                X_seq = X[idx: idx + self.seq_len]
                y = Y[idx + self.seq_len - 1]
                return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

        return DataLoader(SeqDataset(), batch_size=batch_size, shuffle=shuffle)


# =========================================================
#  MODELE GRU
# =========================================================
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# =========================================================
#  TRAINER 
# =========================================================
class GRUTrainer:
    def __init__(self, model, lr=1e-3, device="mps"):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        print("Device:", self.device)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.history = {
            "train_mse": [], "train_rmse": [],
            "valid_mse": [], "valid_rmse": [],
            "test_mse": [],  "test_rmse": [],
            "epoch_time": []
        }

    def _eval(self, loader):
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

    def fit(self, train_loader, valid_loader, test_loader, epochs=120):
        for epoch in range(1, epochs+1):
            t0 = time.time()
            train_mse, train_rmse = self.train_epoch(train_loader)
            valid_mse, valid_rmse = self._eval(valid_loader)
            test_mse,  test_rmse  = self._eval(test_loader)
            dt = time.time() - t0

            self.history["train_mse"].append(train_mse)
            self.history["train_rmse"].append(train_rmse)
            self.history["valid_mse"].append(valid_mse)
            self.history["valid_rmse"].append(valid_rmse)
            self.history["test_mse"].append(test_mse)
            self.history["test_rmse"].append(test_rmse)
            self.history["epoch_time"].append(dt)

            print(
                f"[EPOCH {epoch:03d}] "
                f"Train={train_mse:.6f} | Valid={valid_mse:.6f} | Test={test_mse:.6f} | "
                f"Time={dt:.2f}s"
            )

        return self.history


# =========================================================
#  MAIN
# =========================================================
if __name__ == "__main__":

    df = pd.read_csv("datasetISS_200TLE.csv", sep=";")

    for col in ["dx_km", "dy_km", "dz_km"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # -------- Features --------
    X_cols = [
        "time_utc", "tle_index", "tle_epoch", "dt_since_tle_s",
        "mean_motion", "orbital_speed_km_s", "mean_motion_derivative",
        "altitude_drift_km_per_day", "bstar",
        "inclination_deg", "raan_deg", "eccentricity",
        "arg_perigee_deg", "mean_anomaly_deg", "rev_number",
        "x_sgp4_km", "y_sgp4_km", "z_sgp4_km"
    ]

    df["time_utc"] = pd.to_datetime(df["time_utc"]).astype("int64") / 1e9
    df["tle_epoch"] = pd.to_datetime(df["tle_epoch"]).astype("int64") / 1e9

    def find_col(candidates):
        for cand in candidates:
            for col in df.columns:
                if cand.lower() in col.lower():
                    return col
        raise Exception("Column missing:", candidates)

    x_h_col = find_col(["x_horizons"])
    y_h_col = find_col(["y_horizons"])
    z_h_col = find_col(["z_horizons"])

    df["err_x"] = df[x_h_col] - df["x_sgp4_km"]
    df["err_y"] = df[y_h_col] - df["y_sgp4_km"]
    df["err_z"] = df[z_h_col] - df["z_sgp4_km"]

    X = df[X_cols].values.astype(np.float32)
    Y = df[["err_x","err_y","err_z"]].values.astype(np.float32)

    # -------- Normalisation --------
    X_mean = X.mean(0, keepdims=True)
    X_std  = X.std(0, keepdims=True) + 1e-8
    X = (X - X_mean) / X_std

    seq_len = 128
    batch_size = 32

    # -------- Dataset --------
    dataset = SatelliteSequenceDataset(X, Y, seq_len=seq_len)

    train_loader = dataset.get_loader("train", batch_size=batch_size, shuffle=True)
    valid_loader = dataset.get_loader("valid", batch_size=batch_size, shuffle=False)
    test_loader  = dataset.get_loader("test",  batch_size=batch_size, shuffle=False)

    # -------- EXPERIMENTS --------
    experiments = [
        {"hidden": 64, "layers": 1, "lr": 1e-3},
        {"hidden": 64, "layers": 1, "lr": 1e-4},
        {"hidden": 128, "layers": 3, "lr": 1e-4},
    ]

    input_size = X.shape[1]

    for cfg in experiments:
        print("\n" + "="*70)
        print(f"🚀 GRU EXPERIMENT | hidden={cfg['hidden']} | layers={cfg['layers']} | lr={cfg['lr']}")
        print("="*70)

        model = GRUModel(input_size, hidden_size=cfg["hidden"], num_layers=cfg["layers"])
        trainer = GRUTrainer(model, lr=cfg["lr"])

        hist = trainer.fit(train_loader, valid_loader, test_loader, epochs=80)

        # ---------------------------------------------------------
        # 1) CSV METRICS (TRAIN + VALID + TEST)
        # ---------------------------------------------------------
        df_metrics = pd.DataFrame({
            "Train_MSE": hist["train_mse"],
            "Train_RMSE": hist["train_rmse"],
            "Valid_MSE": hist["valid_mse"],
            "Valid_RMSE": hist["valid_rmse"],
            "Test_MSE":  hist["test_mse"],
            "Test_RMSE": hist["test_rmse"],
            "Time_s":    hist["epoch_time"]
        })

        metrics_path = f"results/Metrics_GRU_H{cfg['hidden']}_L{cfg['layers']}_LR{cfg['lr']}.csv"
        df_metrics.to_csv(metrics_path, index=False)
        print("📄 Saved metrics:", metrics_path)

        # ---------------------------------------------------------
        # 2) CSV ERREURS SUR LE TEST SET
        # ---------------------------------------------------------
        print("Computing test predictions...")

        df_test = df.iloc[dataset.n_train + dataset.n_valid:].reset_index(drop=True)
        X_test = dataset.X_test

        preds = []
        model.eval()

        with torch.no_grad():
            for i in range(seq_len-1, len(X_test)):
                seq = torch.tensor(X_test[i-seq_len+1:i+1], dtype=torch.float32).unsqueeze(0).to(trainer.device)
                preds.append(model(seq).cpu().numpy().squeeze())

        preds = np.array(preds)

        df_sub = df_test.iloc[seq_len-1:].copy().reset_index(drop=True)
        df_sub["err_x_pred"] = preds[:,0]
        df_sub["err_y_pred"] = preds[:,1]
        df_sub["err_z_pred"] = preds[:,2]

        df_errors = df_sub[
            ["x_sgp4_km","y_sgp4_km","z_sgp4_km",
             x_h_col, y_h_col, z_h_col,
             "err_x_pred","err_y_pred","err_z_pred"]
        ]

        out_path = f"results/Erreur_TEST_GRU_H{cfg['hidden']}_L{cfg['layers']}_LR{cfg['lr']}.csv"
        df_errors.to_csv(out_path, index=False)

        print("🎯 Saved test error file:", out_path)

