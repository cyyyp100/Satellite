import numpy as np              # Import de NumPy pour les calculs numériques (tableaux, moyennes, etc.)
import torch                    # Import de PyTorch, le framework de deep learning
import torch.nn as nn           # Raccourci pour le sous-module neural network (couches, modèles, etc.)
from torch.utils.data import Dataset, DataLoader  # Outils pour gérer les datasets et les batchs
import pandas as pd             # Import de pandas pour manipuler les dataframes (CSV, colonnes, etc.)
import time                     # Module pour mesurer les temps d'exécution
import os                       # Module pour interagir avec le système de fichiers

os.makedirs("results", exist_ok=True)  # Crée un dossier "results" si il n'existe pas déjà (pour sauver les CSV)


# =========================================================
#  DATASET — SPLIT PAR TLE
# =========================================================
class SatelliteSequenceDataset(Dataset):
    def __init__(self, X, Y, seq_len, indices):
        self.X = X                            # Matrice des features normalisées (toutes les lignes du dataset)
        self.Y = Y                            # Matrice des cibles (erreurs à prédire, normalisées ici)
        self.seq_len = seq_len                # Longueur de la séquence temporelle qu'on veut en entrée du modèle
        self.indices = np.array(indices)      # Indices (du dataframe) qui appartiennent à ce dataset (train/valid/test)

    def __len__(self):
        # Nombre d'échantillons = nb d'indices moins la taille de la fenêtre (pour construire toutes les séquences possibles)
        return max(0, len(self.indices) - self.seq_len)

    def __getitem__(self, idx):
        # On récupère les indices de la séquence de longueur seq_len à partir de idx
        seq_idxs = self.indices[idx : idx + self.seq_len]
        # L'indice de la cible est le dernier élément de la séquence
        target_idx = self.indices[idx + self.seq_len - 1]

        # Séquence d'entrées X correspondant aux indices de la fenêtre
        X_seq = self.X[seq_idxs]
        # Cible associée à la fin de la séquence
        y = self.Y[target_idx]

        # On retourne la séquence sous forme de tenseur float32, la cible, et l'indice dans le dataframe original
        return (
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            target_idx
        )


# =========================================================
#  TRANSFORMER ORBITAL — VERSION AMÉLIORÉE (SANS POSENC)
# =========================================================
class TransformerOrbital(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=3, dim_feedforward=256):
        super().__init__()  # Appel du constructeur de nn.Module

        # Couche linéaire pour projeter les features d'entrée dans l'espace de dimension d_model
        self.embedding = nn.Linear(input_size, d_model)
        # Normalisation couche par couche des embeddings
        self.emb_norm = nn.LayerNorm(d_model)

        # Définition d'un bloc de TransformerEncoder (une "couche" de Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,              # Dimension interne du modèle (taille des vecteurs)
            nhead=nhead,                  # Nombre de têtes d'attention multi-têtes
            dim_feedforward=dim_feedforward,  # Dimension de la MLP interne du Transformer
            batch_first=True,             # Format des tenseurs: (batch, seq, feature)
            dropout=0.1,                  # Taux de dropout pour régularisation
            activation="gelu"             # Fonction d'activation dans le feed-forward
        )

        # Empilement de plusieurs couches d'encodeur Transformer
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Couche finale linéaire pour projeter la sortie du Transformer sur 3 valeurs (err_x, err_y, err_z)
        self.fc = nn.Linear(d_model, 3)

    def make_causal_mask(self, seq_len, device):
        # Crée un masque triangulaire supérieur (True au-dessus de la diagonale) pour empêcher de voir le futur
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        return mask.to(device)  # Retourne le masque sur le bon device (CPU/GPU/MPS)

    def forward(self, x):
        # x : tensor de forme (batch, seq_len, input_size)

        x = self.embedding(x)    # Projection des features d'entrée vers l'espace d_model
        x = self.emb_norm(x)     # Normalisation des embeddings

        # Construction du masque causal pour l'attention
        mask = self.make_causal_mask(x.size(1), x.device)

        # Passage dans l'encodeur Transformer avec masque causal
        out = self.encoder(x, mask=mask)

        # On récupère uniquement la représentation du dernier pas de temps
        last = out[:, -1, :]
        # Projection linéaire vers 3 sorties (prédiction des 3 erreurs)
        return self.fc(last)


# =========================================================
#  TRAINER AVEC NORMALISATION Y + WARMUP
# =========================================================
class TransformerTrainer:
    def __init__(self, model, Y_mean, Y_std, lr=1e-4, warmup_steps=200, device="mps"):
        # Choix du device : si "mps" dispo (GPU Apple), on l'utilise, sinon CPU
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        print("Using:", self.device)

        self.model = model.to(self.device)          # Envoi du modèle sur le device
        self.loss_fn = nn.MSELoss()                 # Fonction de perte : MSE (Mean Squared Error)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  # Optimiseur AdamW avec lr donné

        # Moyenne et écart-type des cibles, stockés comme tenseurs sur le device
        self.Y_mean = torch.tensor(Y_mean, dtype=torch.float32).to(self.device)
        self.Y_std  = torch.tensor(Y_std, dtype=torch.float32).to(self.device)
        self.warmup_steps = warmup_steps   # Nombre de pas de warmup pour le learning rate
        self.step_count = 0                # Compteur de pas d'entraînement

        # Dictionnaire pour enregistrer l'historique des métriques à chaque epoch
        self.history = {
            "train_mse": [], "train_rmse": [],
            "valid_mse": [], "valid_rmse": [],
            "test_mse":  [], "test_rmse": [],
            "epoch_time": []
        }

    def _normalize_target(self, y):
        # Normalise les cibles en utilisant la moyenne et l'écart type de Y
        return (y - self.Y_mean) / self.Y_std

    def _denormalize(self, y):
        # Opération inverse : repasse des valeurs normalisées à l'échelle originale
        return y * self.Y_std + self.Y_mean

    def _lr_warmup(self, base_lr):
        # Schéma de warmup linéaire du learning rate
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            lr = base_lr * (self.step_count / self.warmup_steps)  # LR augmente linéairement
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr  # Mise à jour du LR dans l'optimiseur

    def _run_epoch(self, loader, train=False):
        losses = []                          # Liste pour enregistrer les pertes de tous les batchs
        self.model.train() if train else self.model.eval()  # Mode train ou eval selon le flag

        # torch.set_grad_enabled active/désactive le calcul des gradients selon train
        with torch.set_grad_enabled(train):
            for X, y, _ in loader:           # Boucle sur les batchs du DataLoader
                X = X.to(self.device)        # Envoi des features sur le device
                y = y.to(self.device)        # Envoi des cibles sur le device

                y_norm = self._normalize_target(y)   # Normalisation des cibles
                pred_norm = self.model(X)            # Prédiction du modèle (déjà en échelle normalisée)

                loss = self.loss_fn(pred_norm, y_norm)  # Calcul de la MSE entre y_norm et pred_norm

                if train:
                    self.optimizer.zero_grad()               # Remise à zéro des gradients
                    loss.backward()                          # Rétropropagation
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Clip des gradients pour stabilité
                    self.optimizer.step()                    # Mise à jour des poids
                    self._lr_warmup(1e-4)                    # Mise à jour du LR via warmup

                losses.append(loss.item())   # Ajout de la valeur scalaire de la perte pour ce batch

        mse = float(np.mean(losses))         # MSE moyenne sur tous les batchs
        rmse = float(np.sqrt(mse))           # RMSE = racine carrée de la MSE
        return mse, rmse                     # Retour des deux métriques

    def predict_test(self, test_loader):
        """Retourne : indices, erreurs prédites dénormalisées"""
        self.model.eval()        # Mode évaluation (pas de dropout, pas de grad)
        preds = []               # Stockage des prédictions
        indices = []             # Stockage des indices d'origine

        with torch.no_grad():    # Désactivation du calcul de gradient
            for X, y, idx in test_loader:
                X = X.to(self.device)        # Envoi des features sur le device

                pred_norm = self.model(X)    # Prédictions normalisées
                pred = self._denormalize(pred_norm).cpu().numpy()  # Dénormalisation + passage en numpy sur CPU

                preds.append(pred)           # Ajout des prédictions du batch
                indices.extend(idx.numpy())  # Ajout des indices correspondants

        # Concaténation de tous les batchs en un seul tableau 2D
        return np.vstack(preds), np.array(indices)


    def fit(self, train_loader, valid_loader, test_loader, epochs=100):
        # Boucle principale d'entraînement sur plusieurs epochs
        for epoch in range(epochs):
            t0 = time.time()  # Temps de début d'epoch

            # Un epoch sur le train, valid et test
            train_mse, train_rmse = self._run_epoch(train_loader, train=True)
            valid_mse, valid_rmse = self._run_epoch(valid_loader, train=False)
            test_mse,  test_rmse  = self._run_epoch(test_loader,  train=False)

            dt = time.time() - t0  # Durée de l'epoch

            # Sauvegarde des métriques dans l'historique
            self.history["train_mse"].append(train_mse)
            self.history["train_rmse"].append(train_rmse)
            self.history["valid_mse"].append(valid_mse)
            self.history["valid_rmse"].append(valid_rmse)
            self.history["test_mse"].append(test_mse)
            self.history["test_rmse"].append(test_rmse)
            self.history["epoch_time"].append(dt)

            print(
                f"[EPOCH {epoch+1:03d}] Train={train_rmse:.4f} | Valid={valid_rmse:.4f} | Test={test_rmse:.4f}"
            )

        # Retourne l'historique complet des métriques
        return self.history



# =========================================================
#  MAIN CODE — SPLIT PAR TLE
# =========================================================
if __name__ == "__main__":

    # Chargement du dataset principal depuis un fichier CSV avec séparateur ';'
    df = pd.read_csv("dataset_hst_sgp4_vs_horizons2.csv", sep=";")

    # Suppression éventuelle des colonnes de différence déjà présentes (dx, dy, dz)
    for c in ["dx_km","dy_km","dz_km"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Conversion des timestamps en secondes depuis l'époque (en float)
    df["time_utc"]  = pd.to_datetime(df["time_utc"]).astype("int64") / 1e9
    df["tle_epoch"] = pd.to_datetime(df["tle_epoch"]).astype("int64") / 1e9

    # Fonction utilitaire pour trouver les colonnes Horizons malgré les variations de noms
    def find_col(names):
        for c in df.columns:                 # Parcourt toutes les colonnes du dataframe
            for n in names:                  # Parcourt tous les motifs possibles
                if n.lower() in c.lower():   # Si le motif apparaît dans le nom de colonne (insensible à la casse)
                    return c                 # On retourne le nom exact de la colonne
        raise Exception("Missing Horizons column")  # Si aucune colonne trouvée, on lève une exception

    # Recherche des colonnes x/y/z venant d'Horizons
    x_h = find_col(["x_horizons"])
    y_h = find_col(["y_horizons"])
    z_h = find_col(["z_horizons"])

    # Calcul des erreurs entre Horizons et SGP4 pour chaque coordonnée
    df["err_x"] = df[x_h] - df["x_sgp4_km"]
    df["err_y"] = df[y_h] - df["y_sgp4_km"]
    df["err_z"] = df[z_h] - df["z_sgp4_km"]

    # Liste des colonnes features (entrées du modèle)
    X_cols = [
        "time_utc","tle_index","tle_epoch","dt_since_tle_s",
        "mean_motion","orbital_speed_km_s","mean_motion_derivative",
        "altitude_drift_km_per_day","bstar","inclination_deg",
        "raan_deg","eccentricity","arg_perigee_deg",
        "mean_anomaly_deg","rev_number",
        "x_sgp4_km","y_sgp4_km","z_sgp4_km"
    ]

    # Construction des matrices numpy X (features) et Y (cibles brutes) en float32
    X = df[X_cols].values.astype(np.float32)
    Y = df[["err_x","err_y","err_z"]].values.astype(np.float32)

    # Normalisation standard de X : (X - moyenne)/std par colonne
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)

    # Calcul de la moyenne et de l'écart type de Y pour la normalisation
    Y_mean = Y.mean(0, keepdims=True)
    Y_std  = Y.std(0, keepdims=True) + 1e-8
    Y_norm = (Y - Y_mean) / Y_std   # Y normalisé

    seq_len = 50   # Longueur de la séquence temporelle utilisée pour l'entraînement

    # SPLIT PAR TLE
    # On partitionne les données en fonction des identifiants TLE pour éviter la fuite entre ensembles
    tle_ids = df["tle_index"].unique()  # Récupère tous les TLE distincts
    n = len(tle_ids)                    # Nombre total de TLE

    # Définition des proportions train/valid/test sur les TLE
    n_train = int(0.60 * n)
    n_valid = int(0.20 * n)
    n_test  = n - n_train - n_valid     # Ce qui reste va au test

    # Découpage des TLE en 3 sous-ensembles
    tle_train = tle_ids[:n_train]
    tle_valid = tle_ids[n_train : n_train+n_valid]
    tle_test  = tle_ids[n_train+n_valid :]

    # Indices des lignes du dataframe appartenant à chaque ensemble (via tle_index)
    idx_train = df.index[df["tle_index"].isin(tle_train)].tolist()
    idx_valid = df.index[df["tle_index"].isin(tle_valid)].tolist()
    idx_test  = df.index[df["tle_index"].isin(tle_test)].tolist()

    # Création des datasets séquentiels pour chaque split
    train_ds = SatelliteSequenceDataset(X, Y_norm, seq_len, idx_train)
    valid_ds = SatelliteSequenceDataset(X, Y_norm, seq_len, idx_valid)
    test_ds  = SatelliteSequenceDataset(X, Y_norm, seq_len, idx_test)

    # DataLoaders : gèrent les batchs et le shuffle
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

    # MODEL
    # Instanciation du modèle Transformer avec les hyperparamètres choisis
    model = TransformerOrbital(
        input_size=X.shape[1],   # Dim. d'entrée = nombre de colonnes de X
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256
    )

    # Création du trainer avec le modèle et les infos de normalisation de Y
    trainer = TransformerTrainer(model, Y_mean, Y_std, lr=1e-4, warmup_steps=200)
    # Entraînement du modèle sur 5 epochs, en évaluant train/valid/test à chaque fois
    hist = trainer.fit(train_loader, valid_loader, test_loader, epochs=5)

    # SAVE METRICS
    # Conversion de l'historique en DataFrame pour sauvegarde
    df_metrics = pd.DataFrame(hist)
    df_metrics.to_csv("results/Metrics_Transformer_Improved.csv", index=False)
    print("Saved metrics")

    # =========================================================
    # EXPORT TEST PREDICTIONS CSV
    # =========================================================
    # Prédictions sur le test set (déjà dénormalisées) + indices correspondants
    preds, pred_indices = trainer.predict_test(test_loader)

    # Copie des lignes du test set dans un nouveau DataFrame
    df_test = df.loc[pred_indices].copy()
    # Ajout des colonnes d'erreurs prédites
    df_test["pred_err_x"] = preds[:,0]
    df_test["pred_err_y"] = preds[:,1]
    df_test["pred_err_z"] = preds[:,2]

    # Coordonnées corrigées = SGP4 + erreur prédite
    df_test["x_corrected"] = df_test["x_sgp4_km"] + df_test["pred_err_x"]
    df_test["y_corrected"] = df_test["y_sgp4_km"] + df_test["pred_err_y"]
    df_test["z_corrected"] = df_test["z_sgp4_km"] + df_test["pred_err_z"]

    # Sauvegarde des prédictions détaillées dans un CSV
    out_path = "results/Test_Predictions.csv"
    df_test.to_csv(out_path, index=False)

    print("Saved detailed test predictions to:", out_path)

    # =========================================================
    #  EXTRA TEST ON A REDUCED DATASET (1000 SAMPLES)
    # =========================================================

    print("\n============================")
    print("🔍 Running extra test dataset")
    print("============================")

    # Load only first 1000 rows of extra dataset
    # Chargement d'un dataset supplémentaire (ISS) et limitation aux 1000 premières lignes
    df_extra = pd.read_csv("datasetISS_200TLE.csv", sep=";").head(1000)

    # Drop irrelevant columns if present
    # Suppression éventuelle des colonnes dx, dy, dz si elles existent
    for c in ["dx_km","dy_km","dz_km"]:
        if c in df_extra.columns:
            df_extra = df_extra.drop(columns=[c])

    # Convert timestamps
    # Conversion des timestamps en secondes depuis l'époque
    df_extra["time_utc"]  = pd.to_datetime(df_extra["time_utc"]).astype("int64") / 1e9
    df_extra["tle_epoch"] = pd.to_datetime(df_extra["tle_epoch"]).astype("int64") / 1e9

    # Horizons columns
    # Fonction utilitaire pour retrouver les colonnes Horizons dans le dataset extra
    def find_col_extra(names):
        for col in df_extra.columns:
            for n in names:
                if n.lower() in col.lower():
                    return col
        raise Exception("Missing Horizons column in extra dataset")

    # Association des colonnes x, y, z Horizons de l'extra dataset
    x_h_e = find_col_extra(["x_horizons"])
    y_h_e = find_col_extra(["y_horizons"])
    z_h_e = find_col_extra(["z_horizons"])

    # Compute true errors
    # Calcul des erreurs "vraies" sur l'extra dataset (Horizons - SGP4)
    df_extra["err_x"] = df_extra[x_h_e] - df_extra["x_sgp4_km"]
    df_extra["err_y"] = df_extra[y_h_e] - df_extra["y_sgp4_km"]
    df_extra["err_z"] = df_extra[z_h_e] - df_extra["z_sgp4_km"]

    # Prepare features
    # Construction des matrices de features et cibles pour l'extra dataset
    X_extra = df_extra[X_cols].values.astype(np.float32)
    Y_extra = df_extra[["err_x","err_y","err_z"]].values.astype(np.float32)

    # Apply SAME normalization as training
    # Application de la même normalisation que pour le dataset d'entraînement
    X_extra = (X_extra - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    Y_extra_norm = (Y_extra - Y_mean) / Y_std

    # Build dataset for sequence inference (same class as train)
    # Création du dataset séquentiel pour l'inférence, utilisant la même classe que pour le train
    idx_extra = df_extra.index.values.tolist()
    extra_ds = SatelliteSequenceDataset(X_extra, Y_extra_norm, seq_len, idx_extra)
    extra_loader = DataLoader(extra_ds, batch_size=32, shuffle=False)

        # Run predictions
    model.eval()
    preds_norm = []   # Stockera les prédictions normalisées du modèle
    true_norm = []    # Stockera les valeurs de vérité terrain normalisées

    with torch.no_grad():
        for batch in extra_loader:

            # --- robust extraction of (X, y) ---
            # On gère le cas où le DataLoader retourne un tuple (X, y, indices)
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    Xb, yb = batch[0], batch[1]   # On récupère X et y
                else:
                    raise ValueError(f"Unexpected batch format: len={len(batch)}")
            else:
                raise ValueError(f"Batch is not a tuple/list: type={type(batch)}")

            # -----------------------------------

            Xb = Xb.to(trainer.device)   # Envoi du batch X sur le même device que le trainer
            pred_b = model(Xb)           # Prédiction du modèle sur ce batch (toujours normalisée)

            preds_norm.append(pred_b.cpu().numpy())  # Ajout des prédictions normalisées converties en numpy
            true_norm.append(yb.numpy())             # Ajout des vraies valeurs normalisées en numpy


    # Empilement de tous les batchs en grands tableaux 2D
    preds_norm = np.vstack(preds_norm)
    true_norm = np.vstack(true_norm)

    # Denormalize predictions
    # Rétablir l'échelle originale pour prédictions et vérité terrain
    preds = preds_norm * Y_std + Y_mean
    true  = true_norm * Y_std + Y_mean

    # Align dataframe rows with predictions (due to sequence window)
    # On enlève les premières lignes (seq_len) pour aligner lignes et prédictions
    df_out = df_extra.iloc[seq_len:].copy()

    # Ajout des colonnes d'erreurs prédites par le modèle
    df_out["pred_err_x"] = preds[:,0]
    df_out["pred_err_y"] = preds[:,1]
    df_out["pred_err_z"] = preds[:,2]

    # Coordonnées corrigées pour l'extra dataset
    df_out["x_corrected"] = df_out["x_sgp4_km"] + df_out["pred_err_x"]
    df_out["y_corrected"] = df_out["y_sgp4_km"] + df_out["pred_err_y"]
    df_out["z_corrected"] = df_out["z_sgp4_km"] + df_out["pred_err_z"]

    # RMSE on external dataset
    # Calcul de la RMSE globale sur le dataset externe (toutes coordonnées confondues)
    rmse_extra = np.sqrt(np.mean((preds - true)**2))
    print(f"\n📊 Extra dataset RMSE: {rmse_extra:.4f} km")

    # Sauvegarde des prédictions sur l'extra dataset dans un CSV séparé
    out_path_extra = "results/Extra_Test_Predictions.csv"
    df_out.to_csv(out_path_extra, index=False)
    print(f"💾 Saved extra test predictions to: {out_path_extra}")

