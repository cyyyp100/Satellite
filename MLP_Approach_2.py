import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import time

os.makedirs("results", exist_ok=True)

# ============================================================
# Dataset 
# ============================================================
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

        print(f"[Dataset] train={len(self.X_train)}, valid={len(self.X_valid)}, test={len(self.X_test)}")

    def get_loader(self, subset, batch_size=256, shuffle=False):
        X = getattr(self, f"X_{subset}")
        Y = getattr(self, f"Y_{subset}")

        class SeqSubset(Dataset):
            def __len__(self2):
                return len(X) - self.seq_len

            def __getitem__(self2, idx):
                X_seq = X[idx: idx + self.seq_len]
                y = Y[idx + self.seq_len - 1]
                return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

        return DataLoader(SeqSubset(), batch_size=batch_size, shuffle=shuffle)


# ============================================================
# MLP Model
# ============================================================
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 256, 128), dropout=0.1):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers.append(nn.Linear(last, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x[:, -1, :])   # on n'utilise que le dernier pas


# ============================================================
# DATASET
# ============================================================
def load_dataset_iss_flat(path_csv="datasetISS_200TLE.csv"):
    df = pd.read_csv(path_csv, sep=";")

    for c in ["dx_km", "dy_km", "dz_km"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    df["time_utc"] = pd.to_datetime(df["time_utc"]).astype("int64") / 1e9
    df["tle_epoch"] = pd.to_datetime(df["tle_epoch"]).astype("int64") / 1e9

    feature_cols = [
        "time_utc", "tle_epoch", "dt_since_tle_s",
        "mean_motion", "orbital_speed_km_s", "mean_motion_derivative",
        "altitude_drift_km_per_day", "bstar",
        "inclination_deg", "raan_deg", "eccentricity",
        "arg_perigee_deg", "mean_anomaly_deg", "rev_number",
        "x_sgp4_km", "y_sgp4_km", "z_sgp4_km",
    ]

    df["err_x"] = df["x_horizons_km"] - df["x_sgp4_km"]
    df["err_y"] = df["y_horizons_km"] - df["y_sgp4_km"]
    df["err_z"] = df["z_horizons_km"] - df["z_sgp4_km"]

    X = df[feature_cols].values.astype(np.float32)
    Y = df[["err_x", "err_y", "err_z"]].values.astype(np.float32)

    X_mean = X.mean(0, keepdims=True)
    X_std  = X.std(0, keepdims=True) + 1e-8
    X = (X - X_mean) / X_std

    return X, Y, X_mean, X_std, df


# ============================================================
# TRAIN
# ============================================================
def train_mlp_on_iss(
    csv_path="datasetISS_200TLE.csv",
    batch_size=256,
    n_epochs=100,
    lr=5e-4,
    seq_len=128,
    hidden_dims=(256, 256, 128)
):
    X, Y, X_mean, X_std, df = load_dataset_iss_flat(csv_path)

    dataset = SatelliteSequenceDataset(X, Y, seq_len=seq_len)

    train_loader = dataset.get_loader("train", batch_size=batch_size, shuffle=True)
    valid_loader = dataset.get_loader("valid", batch_size=batch_size, shuffle=False)
    test_loader  = dataset.get_loader("test",  batch_size=batch_size, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    model = MLPModel(input_dim=X.shape[1], hidden_dims=hidden_dims).to(device)
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_mse": [], "valid_mse": [], "test_mse": [],
        "train_rmse": [], "valid_rmse": [], "test_rmse": [],
        "epoch_time": []
    }

    for epoch in range(1, n_epochs+1):
        t0 = time.time()

        # -------- TRAIN --------
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        train_mse = np.mean(losses)

        # -------- VALID --------
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device)
                losses.append(criterion(model(xb), yb).item())
        valid_mse = np.mean(losses)

        # -------- TEST --------
        losses = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                losses.append(criterion(model(xb), yb).item())
        test_mse = np.mean(losses)

        dt = time.time() - t0

        history["train_mse"].append(train_mse)
        history["valid_mse"].append(valid_mse)
        history["test_mse"].append(test_mse)

        history["train_rmse"].append(np.sqrt(train_mse))
        history["valid_rmse"].append(np.sqrt(valid_mse))
        history["test_rmse"].append(np.sqrt(test_mse))

        history["epoch_time"].append(dt)

        print(f"[MLP][Epoch {epoch:03d}] "
              f"Train={train_mse:.5f} | Valid={valid_mse:.5f} | Test={test_mse:.5f} | {dt:.2f}s")

    # ============================================================
    # 1) CSV METRICS
    # ============================================================
    hidden_str = "-".join(str(h) for h in hidden_dims)
    df_metrics = pd.DataFrame(history)
    df_metrics.index = np.arange(1, len(df_metrics)+1)
    df_metrics.index.name = "Epoch"

    metrics_path = f"results/Metrics_MLP_LR{lr}_HL{hidden_str}.csv"
    df_metrics.to_csv(metrics_path)
    print("Saved metrics:", metrics_path)

    # ============================================================
    # 2) CSV ERREURS — TEST SET
    # ============================================================
    model.eval()

    X_test = dataset.X_test
    df_test = df.iloc[len(dataset.X_train) + len(dataset.X_valid):].reset_index(drop=True)

    preds = []
    with torch.no_grad():
        for i in range(seq_len-1, len(X_test)):
            seq = torch.tensor(X_test[i-seq_len+1:i+1], dtype=torch.float32).unsqueeze(0).to(device)
            preds.append(model(seq).cpu().numpy().squeeze())

    preds = np.array(preds)

    df_sub = df_test.iloc[seq_len-1:].copy().reset_index(drop=True)
    df_sub["err_x_pred"] = preds[:, 0]
    df_sub["err_y_pred"] = preds[:, 1]
    df_sub["err_z_pred"] = preds[:, 2]

    df_errors = df_sub[
        ["x_sgp4_km", "y_sgp4_km", "z_sgp4_km",
         "x_horizons_km", "y_horizons_km", "z_horizons_km",
         "err_x_pred", "err_y_pred", "err_z_pred"]
    ]

    error_path = f"results/Test_Predictions_MLP_LR{lr}_HL{hidden_str}.csv"
    df_errors.to_csv(error_path, index=False)
    print("Saved test error file:", error_path)

    return model, X_mean, X_std


# ============================================================
# MULTI-EXPERIMENTS
# ============================================================
if __name__ == "__main__":
    experiments = [
        {"lr": 1e-3, "hidden_dims": (256, 256, 128)},
        {"lr": 5e-4, "hidden_dims": (256, 256, 128)},
        {"lr": 1e-4, "hidden_dims": (256, 256, 128)},
        {"lr": 1e-4, "hidden_dims": (512, 256, 128)}
    ]

    for i, cfg in enumerate(experiments, start=1):
        print("\n" + "="*70)
        print(f"MLP EXPERIMENT {i} | lr={cfg['lr']} | hidden={cfg['hidden_dims']}")
        print("="*70)

        train_mlp_on_iss(
            csv_path="datasetISS_200TLE.csv",
            batch_size=256,
            n_epochs=100,
            lr=cfg["lr"],
            seq_len=128,
            hidden_dims=cfg["hidden_dims"],
        )
