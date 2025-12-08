import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import pandas as pd
import time  # ⏱ pour mesurer le temps par epoch


class SatelliteSequenceDataset(Dataset):
    def __init__(self, X, Y, seq_len=128, train_ratio=0.8):
        self.X = X
        self.Y = Y
        self.seq_len = seq_len

        assert X.shape[0] == Y.shape[0], "X and Y must align in time"

        split = int(X.shape[0] * train_ratio)
        self.split = split  # 🔹 pour retrouver l’index global validation

        self.X_train, self.Y_train = X[:split], Y[:split]
        self.X_valid, self.Y_valid = X[split:], Y[split:]

        self.train_mode = True


    def set_mode(self, mode="train"):
        self.train_mode = (mode == "train")

    def __len__(self):
        data = self.X_train if self.train_mode else self.X_valid
        return len(data) - self.seq_len

    def __getitem__(self, idx):
        X_data = self.X_train if self.train_mode else self.X_valid
        Y_data = self.Y_train if self.train_mode else self.Y_valid

        X_seq = X_data[idx : idx + self.seq_len]        # (seq_len, features)
        y = Y_data[idx + self.seq_len - 1]              # prédiction du dernier pas

        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        batch_size = x.size(0)

        # état initial du LSTM : h0 et c0
        h_t = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM interne : va dérouler la séquence automatiquement
        out, (h_t, c_t) = self.lstm(x, (h_t, c_t))

        # h_t contient le dernier état caché
        last_hidden_state = h_t[-1]

        # Prédiction finale
        y = self.fc(last_hidden_state)
        return y



class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True


class LSTMTrainer:
    def __init__(self, model, lr=1e-3, device="mps"):
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.history = {
            "train_mse": [],
            "valid_mse": [],
            "train_rmse": [],
            "valid_rmse": [],
            "epoch_time": []
        }

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=20
        )

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

        train_mse = np.mean(losses)
        train_rmse = np.sqrt(train_mse)
        return train_mse, train_rmse

    def eval_epoch(self, loader):
        self.model.eval()
        losses = []

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.loss_fn(pred, y)

                losses.append(loss.item())

        valid_mse = np.mean(losses)
        valid_rmse = np.sqrt(valid_mse)
        return valid_mse, valid_rmse

    def fit(self, train_loader, valid_loader, epochs=100, patience=20):
        es = EarlyStopping(patience=patience)

        for epoch in range(epochs):
            start_time = time.time()  # ⏱ début epoch

            train_mse, train_rmse = self.train_epoch(train_loader)
            valid_mse, valid_rmse = self.eval_epoch(valid_loader)

            epoch_time = time.time() - start_time

            self.history["train_mse"].append(train_mse)
            self.history["valid_mse"].append(valid_mse)
            self.history["train_rmse"].append(train_rmse)
            self.history["valid_rmse"].append(valid_rmse)
            self.history["epoch_time"].append(epoch_time)

            # 🔹 scheduler sur la loss de validation
            self.scheduler.step(valid_mse)

            print(
                f"[EPOCH {epoch+1:03d}] "
                f"train_mse={train_mse:.6f} | "
                f"valid_mse={valid_mse:.6f} | "
                f"valid_rmse={valid_rmse:.6f} | "
                f"temps_calcul={epoch_time:.3f}s"
            )

            es(valid_mse)
            if es.early_stop:
                print("⛔ Early stopping triggered.")
                break

        return self.history


class ExperimentRunner:
    def __init__(self):
        self.results = []

    def run(self, name, model, trainer, train_loader, valid_loader, epochs=120, patience=20):
        print(f"\n🚀 Running experiment: {name}")
        history = trainer.fit(train_loader, valid_loader, epochs, patience)
        self.results.append({"name": name, "history": history})
        return history

    def plot(self):
        plt.figure(figsize=(14,6))

        for res in self.results:
            plt.plot(res["history"]["valid_rmse"], label=res["name"])

        plt.title("Comparaison des RMSE Validation (LSTM)")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":

    # ---------------------
    # Load real dataset
    # ---------------------
    csv_path = "datasetISS_200TLE.csv"  # adapter le chemin si besoin
    df = pd.read_csv(csv_path, sep=";")

    # Supprimer dx, dy, dz si présents
    for col in ["dx_km", "dy_km", "dz_km"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Colonnes d'entrée X
    X_cols = [
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

    # Conversion des dates en timestamps numériques
    if "time_utc" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time_utc"]).astype("int64") / 1e9  # secondes
    if "tle_epoch" in df.columns:
        df["tle_epoch"] = pd.to_datetime(df["tle_epoch"]).astype("int64") / 1e9

    # Debug: afficher les colonnes disponibles pour vérifier les noms réels
    print("Colonnes du CSV :", list(df.columns))

    # Petite fonction utilitaire pour retrouver une colonne Horizons même si le nom varie un peu
    def find_col(candidates):
        cols_lower = {c.lower().strip(): c for c in df.columns}
        for cand in candidates:
            key = cand.lower().strip()
            if key in cols_lower:
                return cols_lower[key]
        # Ultime recours : chercher en 'contains'
        for key, original in cols_lower.items():
            for cand in candidates:
                if cand.lower().strip() in key:
                    return original
        raise KeyError(f"Aucune colonne trouvée parmi {candidates} dans {df.columns}")

    # On essaie plusieurs variantes possibles des noms de colonnes Horizons
    x_h_col = find_col(["x_horizons_km", "x_horizon_km", "x_horizons"])
    y_h_col = find_col(["y_horizons_km", "y_horizon_km", "y_horizons"])
    z_h_col = find_col(["z_horizons_km", "z_horizon_km", "z_horizons"])

    # Calcul des erreurs SGP4 -> Horizons
    df["err_x"] = df[x_h_col] - df["x_sgp4_km"]
    df["err_y"] = df[y_h_col] - df["y_sgp4_km"]
    df["err_z"] = df[z_h_col] - df["z_sgp4_km"]

    # Matrices numpy
    X = df[X_cols].values.astype(np.float32)
    Y = df[["err_x", "err_y", "err_z"]].values.astype(np.float32)

    # Normalisation simple (z-score) de X
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - X_mean) / X_std

    seq_len = 128
    batch_size = 32

    dataset = SatelliteSequenceDataset(X, Y, seq_len=seq_len)

    train_set, valid_set = dataset, dataset  # same object, mode changes
    train_set.set_mode("train")
    valid_set.set_mode("valid")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    # ---------------------
    # RUN EXPERIMENTS
    # ---------------------
    runner = ExperimentRunner()

    # Experiment 1
    model1 = LSTMModel(input_size=18, hidden_size=64, num_layers=1)
    trainer1 = LSTMTrainer(model1, lr=1e-4)

    runner.run("LSTM_64_hidden_lr1e-4", model1, trainer1, train_loader, valid_loader)

    '''# Experiment 2
    model2 = LSTMModel(input_size=18, hidden_size=128, num_layers=2)
    trainer2 = LSTMTrainer(model2, lr=5e-4)

    runner.run("LSTM_128_hidden_lr5e-4", model2, trainer2, train_loader, valid_loader)
    '''
    # ---------------------
    # Plot results (RMSE)
    # ---------------------
    runner.plot()

    # ---------------------
    # Comparaison SGP4 vs Horizons vs modèle (sur la validation) — comme pour le GRU
    # ---------------------
    model1.eval()
    device = trainer1.device

    valid_set.set_mode("valid")
    split = dataset.split
    n_valid = dataset.X_valid.shape[0]

    all_idx = []
    pred_err_list = []

    with torch.no_grad():
        for i in range(n_valid - seq_len):
            X_seq, _ = valid_set[i]
            X_seq = X_seq.unsqueeze(0).to(device)
            pred_err = model1(X_seq).cpu().numpy()[0]   # (3,)
            pred_err_list.append(pred_err)

            idx_global = split + i + seq_len - 1
            all_idx.append(idx_global)

    all_idx = np.array(all_idx)
    pred_err = np.array(pred_err_list)

    sgp4_pos = df.loc[all_idx, ["x_sgp4_km", "y_sgp4_km", "z_sgp4_km"]].values
    horizons_pos = df.loc[all_idx, [x_h_col, y_h_col, z_h_col]].values

    model_pos = sgp4_pos + pred_err

    err_sgp4 = np.linalg.norm(horizons_pos - sgp4_pos, axis=1)
    err_model = np.linalg.norm(horizons_pos - model_pos, axis=1)

    plt.figure(figsize=(12, 5))
    plt.plot(err_sgp4, label="Erreur 3D SGP4 → Horizons")
    plt.plot(err_model, label="Erreur 3D Modèle LSTM (SGP4 + correction)", alpha=0.8)
    plt.xlabel("Indice dans la série de validation")
    plt.ylabel("Erreur 3D (km)")
    plt.title("Comparaison des erreurs 3D : SGP4 vs modèle LSTM corrigé")
    plt.legend()
    plt.grid(True)
    plt.show()