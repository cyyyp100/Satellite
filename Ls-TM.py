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
    - X_seq : (seq_len, input_dim)
    - Y     : (3,)  (err_x, err_y, err_z) au dernier instant de la séquence
    """
    def __init__(self, X_seq: np.ndarray, Y_vec: np.ndarray):
        """
        X_seq : shape (N_seq, seq_len, input_dim)
        Y_vec : shape (N_seq, 3)
        """
        assert X_seq.shape[0] == Y_vec.shape[0], "Mismatch N_seq entre X_seq et Y_vec"
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.Y_vec = torch.tensor(Y_vec, dtype=torch.float32)

    def __len__(self):
        return self.X_seq.shape[0]

    def __getitem__(self, idx):
        return self.X_seq[idx], self.Y_vec[idx]


# ============================================================
# Modèle LSTM pour correction SGP4
# ============================================================
class LSTMCorrectionModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 bidirectional: bool = False, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_dim, 3)  # prédire (err_x, err_y, err_z)

    def forward(self, x):
        """
        x : (batch_size, seq_len, input_dim)
        """
        out, _ = self.lstm(x)           # out : (batch_size, seq_len, lstm_out_dim)
        last = out[:, -1, :]            # prendre le dernier pas de temps
        pred = self.fc(last)            # (batch_size, 3)
        return pred


# ============================================================
# Construction des séquences glissantes
# ============================================================
def build_sequences(X: np.ndarray, Y: np.ndarray, seq_len: int):
    """
    X : (N, F)
    Y : (N, 3)
    Retour :
        X_seq : (N_seq, seq_len, F)
        Y_seq : (N_seq, 3)  (erreur au dernier instant de la séquence)
    """
    N = X.shape[0]
    if N < seq_len:
        raise ValueError(f"Nombre d'échantillons ({N}) < seq_len ({seq_len})")

    N_seq = N - seq_len + 1
    F = X.shape[1]

    X_seq = np.zeros((N_seq, seq_len, F), dtype=np.float32)
    Y_seq = np.zeros((N_seq, 3), dtype=np.float32)

    for i in range(N_seq):
        X_seq[i] = X[i:i + seq_len]
        Y_seq[i] = Y[i + seq_len - 1]  # cible = erreur au dernier instant

    return X_seq, Y_seq


# ============================================================
# Chargement des données : datasetISS_200TLE.csv
# ============================================================
def load_dataset_iss(path_csv: str = "datasetISS_200TLE.csv", seq_len: int = 128):
    # Lecture CSV (séparateur ";")
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

    # Features d'entrée (mêmes que pour le GRU)
    feature_cols = [
        "time_utc",
        "tle_index",
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

    # Séquences
    X_seq, Y_seq = build_sequences(X_all, Y_all, seq_len=seq_len)

    return X_seq, Y_seq, X_mean, X_std


# ============================================================
# Entraînement LSTM sur ISS
# ============================================================
def train_lstm_on_iss(
    csv_path: str = "datasetISS_200TLE.csv",
    seq_len: int = 128,
    batch_size: int = 32,
    n_epochs: int = 100,
    lr: float = 5e-4,
):
    # Charger les données
    X_seq, Y_seq, X_mean, X_std = load_dataset_iss(csv_path, seq_len)
    N_seq = X_seq.shape[0]

    # Split temporel 80% / 20% (comme pour le GRU)
    split = int(0.8 * N_seq)
    X_train, Y_train = X_seq[:split], Y_seq[:split]
    X_val, Y_val = X_seq[split:], Y_seq[split:]

    train_ds = SatelliteSequenceDataset(X_train, Y_train)
    val_ds = SatelliteSequenceDataset(X_val, Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Device (MPS sur Mac si dispo)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    input_dim = X_seq.shape[2]
    model = LSTMCorrectionModel(input_dim=input_dim, hidden_dim=128, num_layers=2, bidirectional=False)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def rmse_from_mse(mse: float) -> float:
        return float(np.sqrt(mse))

    # Boucle d'entraînement
    for epoch in range(1, n_epochs + 1):
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

        # Validation
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
            f"[LSTM][Epoch {epoch:03d}] "
            f"Train MSE={train_mse:.6f} | Val MSE={val_mse:.6f} | "
            f"Train RMSE={train_rmse:.6f} | Val RMSE={val_rmse:.6f}"
        )

    return model, X_mean, X_std


if __name__ == "__main__":
    # Entraînement de l'LSTM sur le dataset ISS SGP4 vs Horizons
    train_lstm_on_iss()