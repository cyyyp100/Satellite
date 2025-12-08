import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time
import os

os.makedirs("results", exist_ok=True)

# =========================================================
#  DATASET basé sur SPLIT PAR TLE
# =========================================================
class SatelliteSequenceDataset(Dataset):
    def __init__(self, X, Y, seq_len=50):
        self.X = X
        self.Y = Y
        self.seq_len = seq_len
        self.indices = None

    def set_indices(self, indices):
        self.indices = np.array(indices)

    def __len__(self):
        return len(self.indices) - self.seq_len

    def __getitem__(self, idx):
        real_idx    = self.indices[idx : idx + self.seq_len]
        target_idx  = self.indices[idx + self.seq_len - 1]

        X_seq = self.X[real_idx]
        y     = self.Y[target_idx]

        return (
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            target_idx
        )


# =========================================================
#  MODELE LSTM
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
#  EARLY STOPPING
# =========================================================
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-5):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = np.inf
        self.early_stop = False

    def __call__(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True


# =========================================================
#  TRAINER
# =========================================================
class LSTMTrainer:
    def __init__(self, model, lr=1e-3, device="mps"):
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.history = {
            "train_mse": [], "train_rmse": [],
            "valid_mse": [], "valid_rmse": [],
            "test_mse": [],  "test_rmse": [],
            "epoch_time": []
        }

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=20
        )

    def _run_epoch(self, loader, train=False):
        losses = []

        self.model.train() if train else self.model.eval()

        with torch.set_grad_enabled(train):
            for X, y, _ in loader:
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                loss = self.loss_fn(pred, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                losses.append(loss.item())

        mse = np.mean(losses)
        rmse = np.sqrt(mse)
        return mse, rmse


    def predict_test(self, loader):
        self.model.eval()
        preds = []
        indices = []

        with torch.no_grad():
            for X, y, idx in loader:
                X = X.to(self.device)
                pred = self.model(X).cpu().numpy()

                preds.append(pred)
                indices.extend(idx.numpy())

        return np.vstack(preds), np.array(indices)

    def fit(self, train_loader, valid_loader, test_loader, epochs=150, patience=20):
        es = EarlyStopping(patience)

        for epoch in range(epochs):
            t0 = time.time()

            train_mse, train_rmse = self._run_epoch(train_loader, train=True)
            valid_mse, valid_rmse = self._run_epoch(valid_loader, train=False)
            test_mse,  test_rmse  = self._run_epoch(test_loader,  train=False)

            dt = time.time() - t0

            self.history["train_mse"].append(train_mse)
            self.history["train_rmse"].append(train_rmse)
            self.history["valid_mse"].append(valid_mse)
            self.history["valid_rmse"].append(valid_rmse)
            self.history["test_mse"].append(test_mse)
            self.history["test_rmse"].append(test_rmse)
            self.history["epoch_time"].append(dt)

            self.scheduler.step(valid_mse)

            print(
                f"[EPOCH {epoch+1:03d}] "
                f"Train={train_mse:.4f} | Valid={valid_mse:.4f} | Test={test_mse:.4f} | "
                f"RMSE_valid={valid_rmse:.4f}"
            )

            es(valid_mse)
            if es.early_stop:
                print("⛔ EARLY STOPPING")
                break

        return self.history


# =========================================================
#  MAIN 
# =========================================================
if __name__ == "__main__":

    df = pd.read_csv("datasetISS_200TLE.csv", sep=";")

    for col in ["dx_km","dy_km","dz_km"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df["time_utc"]  = pd.to_datetime(df["time_utc"]).astype("int64") / 1e9
    df["tle_epoch"] = pd.to_datetime(df["tle_epoch"]).astype("int64") / 1e9

    def find_col(cands):
        for col in df.columns:
            for c in cands:
                if c.lower() in col.lower():
                    return col
        raise ValueError(cands)

    x_h = find_col(["x_horizons"])
    y_h = find_col(["y_horizons"])
    z_h = find_col(["z_horizons"])

    df["err_x"] = df[x_h] - df["x_sgp4_km"]
    df["err_y"] = df[y_h] - df["y_sgp4_km"]
    df["err_z"] = df[z_h] - df["z_sgp4_km"]

    X_cols = [
        "time_utc","tle_index","tle_epoch","dt_since_tle_s",
        "mean_motion","orbital_speed_km_s","mean_motion_derivative",
        "altitude_drift_km_per_day","bstar","inclination_deg",
        "raan_deg","eccentricity","arg_perigee_deg",
        "mean_anomaly_deg","rev_number",
        "x_sgp4_km","y_sgp4_km","z_sgp4_km"
    ]

    X = df[X_cols].values.astype(np.float32)
    Y = df[["err_x","err_y","err_z"]].values.astype(np.float32)

    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)

    # =====================================================
    # SPLIT PAR TLE INDEX
    # =====================================================
    tle_ids = df["tle_index"].unique()
    n = len(tle_ids)

    n_train = int(0.60 * n)
    n_valid = int(0.20 * n)
    n_test  = n - n_train - n_valid

    tle_train = tle_ids[:n_train]
    tle_valid = tle_ids[n_train : n_train+n_valid]
    tle_test  = tle_ids[n_train+n_valid :]

    idx_train = df.index[df["tle_index"].isin(tle_train)].tolist()
    idx_valid = df.index[df["tle_index"].isin(tle_valid)].tolist()
    idx_test  = df.index[df["tle_index"].isin(tle_test)].tolist()

    print(f"SPLIT → Train={len(idx_train)}, Valid={len(idx_valid)}, Test={len(idx_test)}")

    dataset_train = SatelliteSequenceDataset(X, Y, seq_len=50)
    dataset_valid = SatelliteSequenceDataset(X, Y, seq_len=50)
    dataset_test  = SatelliteSequenceDataset(X, Y, seq_len=50)

    dataset_train.set_indices(idx_train)
    dataset_valid.set_indices(idx_valid)
    dataset_test.set_indices(idx_test)

    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=32, shuffle=False)
    test_loader  = DataLoader(dataset_test,  batch_size=32, shuffle=False)

    # =====================================================
    # EXPÉRIMENTS
    # =====================================================
    experiments = [
        {"hidden_size": 64,  "num_layers": 1, "lr": 1e-3},
        {"hidden_size": 64,  "num_layers": 1, "lr": 1e-4},
        {"hidden_size": 64,  "num_layers": 3, "lr": 1e-4},
        {"hidden_size": 128, "num_layers": 1, "lr": 1e-4},
        {"hidden_size": 128, "num_layers": 3, "lr": 1e-4},
        {"hidden_size": 128, "num_layers": 3, "lr": 5e-5},
    ]

    input_size = X.shape[1]

    for cfg in experiments:

        print("\n======================================")
        print(f"🏁 EXPERIMENT H={cfg['hidden_size']} L={cfg['num_layers']}")
        print("======================================")

        model = LSTMModel(input_size, cfg["hidden_size"], cfg["num_layers"])
        trainer = LSTMTrainer(model, lr=cfg["lr"])

        history = trainer.fit(train_loader, valid_loader, test_loader, epochs=120)

        df_metrics = pd.DataFrame(history)
        out_path = f"results/Metrics_LSTM_H{cfg['hidden_size']}_L{cfg['num_layers']}_seqlen50.csv"
        df_metrics.to_csv(out_path, index=False)
        print(f"💾 Saved metrics: {out_path}")

        # =========================================================
        #  EXPORT TEST PREDICTIONS CSV
        # =========================================================
        preds, pred_indices = trainer.predict_test(test_loader)

        df_test = df.loc[pred_indices].copy()
        df_test["pred_err_x"] = preds[:,0]
        df_test["pred_err_y"] = preds[:,1]
        df_test["pred_err_z"] = preds[:,2]

        df_test["x_corrected"] = df_test["x_sgp4_km"] + df_test["pred_err_x"]
        df_test["y_corrected"] = df_test["y_sgp4_km"] + df_test["pred_err_y"]
        df_test["z_corrected"] = df_test["z_sgp4_km"] + df_test["pred_err_z"]

        out_path2 = f"results/Test_Predictions_LSTM_H{cfg['hidden_size']}_L{cfg['num_layers']}_seqlen50.csv"
        df_test.to_csv(out_path2, index=False)
        print(f"📄 Saved corrected predictions: {out_path2}")
