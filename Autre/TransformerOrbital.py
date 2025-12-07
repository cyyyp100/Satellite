import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time
import os

os.makedirs("results", exist_ok=True)


# =========================================================
#  DATASET — SPLIT PAR TLE
# =========================================================
class SatelliteSequenceDataset(Dataset):
    def __init__(self, X, Y, seq_len, indices):
        self.X = X
        self.Y = Y
        self.seq_len = seq_len
        self.indices = np.array(indices)

    def __len__(self):
        return max(0, len(self.indices) - self.seq_len)

    def __getitem__(self, idx):
        seq_idxs = self.indices[idx : idx + self.seq_len]
        target_idx = self.indices[idx + self.seq_len - 1]

        X_seq = self.X[seq_idxs]
        y = self.Y[target_idx]

        return (
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )


# =========================================================
#  TRANSFORMER ORBITAL — VERSION AMÉLIORÉE (SANS POSENC)
# =========================================================
class TransformerOrbital(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=3, dim_feedforward=256):
        super().__init__()

        self.embedding = nn.Linear(input_size, d_model)
        self.emb_norm = nn.LayerNorm(d_model)

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

    def make_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        return mask.to(device)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.embedding(x)
        x = self.emb_norm(x)

        mask = self.make_causal_mask(x.size(1), x.device)

        out = self.encoder(x, mask=mask)

        last = out[:, -1, :]  # dernier pas temporel
        return self.fc(last)


# =========================================================
#  TRAINER AVEC NORMALISATION Y + WARMUP
# =========================================================
class TransformerTrainer:
    def __init__(self, model, Y_mean, Y_std, lr=1e-4, warmup_steps=200, device="mps"):
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        print("Using:", self.device)

        self.model = model.to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        self.Y_mean = torch.tensor(Y_mean, dtype=torch.float32).to(self.device)
        self.Y_std  = torch.tensor(Y_std, dtype=torch.float32).to(self.device)
        self.warmup_steps = warmup_steps
        self.step_count = 0

        self.history = {
            "train_mse": [], "train_rmse": [],
            "valid_mse": [], "valid_rmse": [],
            "test_mse":  [], "test_rmse": [],
            "epoch_time": []
        }

    def _normalize_target(self, y):
        return (y - self.Y_mean) / self.Y_std

    def _denormalize(self, y):
        return y * self.Y_std + self.Y_mean

    def _lr_warmup(self, base_lr):
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            lr = base_lr * (self.step_count / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    def _run_epoch(self, loader, train=False):
        losses = []

        self.model.train() if train else self.model.eval()

        with torch.set_grad_enabled(train):
            for X, y in loader:
                X = X.to(self.device)
                y = y.to(self.device)

                y_norm = self._normalize_target(y)

                pred_norm = self.model(X)

                loss = self.loss_fn(pred_norm, y_norm)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self._lr_warmup(1e-4)

                losses.append(loss.item())

        mse = float(np.mean(losses))
        rmse = float(np.sqrt(mse))
        return mse, rmse

    def fit(self, train_loader, valid_loader, test_loader, epochs=100):
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

            print(
                f"[EPOCH {epoch+1:03d}] "
                f"Train={train_rmse:.4f} | Valid={valid_rmse:.4f} | Test={test_rmse:.4f}"
            )

        return self.history


# =========================================================
#  MAIN CODE — SPLIT PAR TLE
# =========================================================
if __name__ == "__main__":

    df = pd.read_csv("dataset_hst_sgp4_vs_horizons2.csv", sep=";")

    for c in ["dx_km","dy_km","dz_km"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    df["time_utc"]  = pd.to_datetime(df["time_utc"]).astype("int64") / 1e9
    df["tle_epoch"] = pd.to_datetime(df["tle_epoch"]).astype("int64") / 1e9

    def find_col(names):
        for c in df.columns:
            for n in names:
                if n.lower() in c.lower():
                    return c
        raise Exception("Missing Horizons column")

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

    # Normalisation Y (IMPORTANT)
    Y_mean = Y.mean(0, keepdims=True)
    Y_std  = Y.std(0, keepdims=True) + 1e-8
    Y_norm = (Y - Y_mean) / Y_std

    seq_len = 50

    # SPLIT PAR TLE
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

    # datasets
    train_ds = SatelliteSequenceDataset(X, Y_norm, seq_len, idx_train)
    valid_ds = SatelliteSequenceDataset(X, Y_norm, seq_len, idx_valid)
    test_ds  = SatelliteSequenceDataset(X, Y_norm, seq_len, idx_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

    # MODEL
    model = TransformerOrbital(
        input_size=X.shape[1],
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256
    )

    trainer = TransformerTrainer(model, Y_mean, Y_std, lr=1e-4, warmup_steps=200)
    hist = trainer.fit(train_loader, valid_loader, test_loader, epochs=20)

    # SAVE METRICS
    df_metrics = pd.DataFrame(hist)
    df_metrics.to_csv("results/Metrics_Transformer_Improved.csv", index=False)

    print("\nSaved metrics to results/Metrics_Transformer_Improved.csv")

        # =========================================================
    #  EXTRA TEST ON A NEW DATASET (NO TRAINING)
    # =========================================================

    print("\n============================")
    print("🔍 Running extra test dataset")
    print("============================")

    # Load your extra dataset (modify the file name if needed)
    df_extra = pd.read_csv("extra_dataset.csv", sep=";")

    # Drop irrelevant columns if present
    for c in ["dx_km","dy_km","dz_km"]:
        if c in df_extra.columns:
            df_extra = df_extra.drop(columns=[c])

    # Convert timestamps
    df_extra["time_utc"]  = pd.to_datetime(df_extra["time_utc"]).astype("int64") / 1e9
    df_extra["tle_epoch"] = pd.to_datetime(df_extra["tle_epoch"]).astype("int64") / 1e9

    # Find Horizons columns
    def find_col_extra(names):
        for col in df_extra.columns:
            for n in names:
                if n.lower() in col.lower():
                    return col
        raise Exception("Missing Horizons column in extra dataset")

    x_h_e = find_col_extra(["x_horizons"])
    y_h_e = find_col_extra(["y_horizons"])
    z_h_e = find_col_extra(["z_horizons"])

    # Compute true errors
    df_extra["err_x"] = df_extra[x_h_e] - df_extra["x_sgp4_km"]
    df_extra["err_y"] = df_extra[y_h_e] - df_extra["y_sgp4_km"]
    df_extra["err_z"] = df_extra[z_h_e] - df_extra["z_sgp4_km"]

    # Prepare input features
    X_extra = df_extra[X_cols].values.astype(np.float32)
    Y_extra = df_extra[["err_x","err_y","err_z"]].values.astype(np.float32)

    # Apply SAME NORMALIZATION as training
    X_extra = (X_extra - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    Y_extra_norm = (Y_extra - Y_mean) / Y_std

    # Build dataset indices (full sequence)
    idx_extra = df_extra.index.values.tolist()

    extra_ds = SatelliteSequenceDataset(X_extra, Y_extra_norm, seq_len, idx_extra)
    extra_loader = DataLoader(extra_ds, batch_size=32, shuffle=False)

    # Run model in eval mode
    model.eval()
    preds_norm = []
    true_norm = []

    with torch.no_grad():
        for Xb, yb in extra_loader:
            Xb = Xb.to(trainer.device)
            pred_b = model(Xb)
            preds_norm.append(pred_b.cpu().numpy())
            true_norm.append(yb.numpy())

    preds_norm = np.vstack(preds_norm)
    true_norm = np.vstack(true_norm)

    # Denormalize predictions
    preds = preds_norm * Y_std + Y_mean
    true  = true_norm * Y_std + Y_mean

    # Save predictions
    df_out = df_extra.iloc[seq_len:].copy()  # align indices

    df_out["pred_err_x"] = preds[:,0]
    df_out["pred_err_y"] = preds[:,1]
    df_out["pred_err_z"] = preds[:,2]

    # Corrected positions
    df_out["x_corrected"] = df_out["x_sgp4_km"] + df_out["pred_err_x"]
    df_out["y_corrected"] = df_out["y_sgp4_km"] + df_out["pred_err_y"]
    df_out["z_corrected"] = df_out["z_sgp4_km"] + df_out["pred_err_z"]

    # Compute RMSE on extra dataset
    rmse_extra = np.sqrt(np.mean((preds - true)**2))
    print(f"\n📊 Extra dataset RMSE: {rmse_extra:.4f} km")

    # Save CSV
    out_path_extra = "results/Extra_Test_Predictions.csv"
    df_out.to_csv(out_path_extra, index=False)
    print(f"💾 Saved extra test predictions to: {out_path_extra}")

