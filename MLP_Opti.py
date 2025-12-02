import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Dataset séquentiel pour correction SGP4 -> Horizons
# ============================================================
class SatelliteSequenceDataset(Dataset):
    """
    Dataset : séquences temporelles (X_seq) -> vecteur d'erreur (Y)
    - X_seq : (seq_len, features)
    - Y     : (3,) (err_x, err_y, err_z) au dernier pas de la séquence
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, seq_len: int = 128, train_ratio: float = 0.8):
        self.X = X
        self.Y = Y
        self.seq_len = seq_len

        assert X.shape[0] == Y.shape[0], "X and Y must align in time"

        split = int(X.shape[0] * train_ratio)
        self.X_train, self.Y_train = X[:split], Y[:split]
        self.X_valid, self.Y_valid = X[split:], Y[split:]

        self.train_mode = True

    def set_mode(self, mode: str = "train"):
        self.train_mode = (mode == "train")

    def __len__(self):
        data = self.X_train if self.train_mode else self.X_valid
        return len(data) - self.seq_len

    def __getitem__(self, idx):
        X_data = self.X_train if self.train_mode else self.X_valid
        Y_data = self.Y_train if self.train_mode else self.Y_valid

        X_seq = X_data[idx: idx + self.seq_len]       # (seq_len, features)
        y = Y_data[idx + self.seq_len - 1]           # prédiction au dernier pas

        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ============================================================
# MLP pour correction SGP4
# ============================================================
class MLPModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(256, 256, 128), dropout: float = 0.1):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = h
        layers.append(nn.Linear(last_dim, 3))  # prédire (err_x, err_y, err_z)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x : (batch_size, seq_len, input_dim)
        """
        # Utilise uniquement le dernier pas de temps comme entrée au MLP
        x_last = x[:, -1, :]  # (batch_size, input_dim)
        return self.net(x_last)


# ============================================================
# Chargement des données : datasetISS_200TLE.csv (version "flat")
# ============================================================
def load_dataset_iss_flat(path_csv: str = "datasetISS_200TLE.csv"):
    # Lecture CSV
    df = pd.read_csv(path_csv, sep=";")

    # Suppression éventuelle des colonnes dx,dy,dz
    for col in ["dx_km", "dy_km", "dz_km"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Conversion des dates en timestamps (secondes)
    if "time_utc" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time_utc"]).astype("int64") / 1e9
    if "tle_epoch" in df.columns:
        df["tle_epoch"] = pd.to_datetime(df["tle_epoch"]).astype("int64") / 1e9

    # Features d'entrée (mêmes que GRU / LSTM)
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

    X_all = df[feature_cols].values.astype(np.float32)
    Y_all = df[["err_x", "err_y", "err_z"]].values.astype(np.float32)

    # Normalisation de X (z-score)
    X_mean = X_all.mean(axis=0, keepdims=True)
    X_std = X_all.std(axis=0, keepdims=True) + 1e-8
    X_all = (X_all - X_mean) / X_std

    return X_all, Y_all, X_mean, X_std


# ============================================================
# Entraînement MLP sur ISS
# ============================================================
def train_mlp_on_iss(
    csv_path: str = "datasetISS_200TLE.csv",
    batch_size: int = 256,
    n_epochs: int = 100,
    lr: float = 5e-4,
    seq_len: int = 128,
):
    # Charger les données (features "flat" X_all, erreurs Y_all)
    X_all, Y_all, X_mean, X_std = load_dataset_iss_flat(csv_path)

    # Dataset séquentiel avec split temporel interne 80% / 20%
    train_ds = SatelliteSequenceDataset(X_all, Y_all, seq_len=seq_len, train_ratio=0.8)
    val_ds = SatelliteSequenceDataset(X_all, Y_all, seq_len=seq_len, train_ratio=0.8)
    val_ds.set_mode("valid")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Device (MPS sur Mac si dispo)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device for MLP: {device}")

    input_dim = X_all.shape[1]
    model = MLPModel(input_dim=input_dim, hidden_dims=(256, 256, 128), dropout=0.1)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def rmse_from_mse(mse: float) -> float:
        return float(np.sqrt(mse))

    for epoch in range(1, n_epochs + 1):
        # ---------- Entraînement ----------
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * xb.size(0)
            n_train += xb.size(0)

        train_mse = train_loss_sum / n_train
        train_rmse = rmse_from_mse(train_mse)

        # ---------- Validation ----------
        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss_sum += loss.item() * xb.size(0)
                n_val += xb.size(0)

        val_mse = val_loss_sum / n_val
        val_rmse = rmse_from_mse(val_mse)

        print(
            f"[MLP][Epoch {epoch:03d}] "
            f"Train MSE={train_mse:.6f} | Val MSE={val_mse:.6f} | "
            f"Train RMSE={train_rmse:.6f} | Val RMSE={val_rmse:.6f}"
        )

    return model, X_mean, X_std


if __name__ == "__main__":
    # Entraînement du MLP sur le dataset ISS SGP4 vs Horizons
    train_mlp_on_iss()