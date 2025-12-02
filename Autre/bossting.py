import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import pandas as pd


# ======================================================================
#  DATASET
# ======================================================================
class SatelliteDataset:
    """
    Manage temporal dataset built from X (features) and Y (true error vectors).
    X has shape (T, F)
    Y has shape (T, 3) (ex: error = Horizons - SGP4)
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, train_ratio: float = 0.8):
        assert X.shape[0] == Y.shape[0], "X and Y must have same temporal length."

        self.X = X
        self.Y = Y
        self.train_ratio = train_ratio
        self._train_test_split()

    def _train_test_split(self):
        split = int(self.X.shape[0] * self.train_ratio)
        self.X_train = self.X[:split]
        self.Y_train = self.Y[:split]

        self.X_valid = self.X[split:]
        self.Y_valid = self.Y[split:]

    def get_train(self):
        return self.X_train, self.Y_train

    def get_valid(self):
        return self.X_valid, self.Y_valid


# ======================================================================
#  TRAINER
# ======================================================================
class XGBoostTrainer:
    """
    Train an XGBoost model to predict satellite position error (3D vector).
    """
    def __init__(self, params: Dict[str, Any], num_rounds: int = 500, early_stopping: int = 20):
        self.params = params
        self.num_rounds = num_rounds
        self.early_stopping = early_stopping
        self.model = None
        self.evals_result = {}

    def train(self, X_train, Y_train, X_valid, Y_valid):
        """
        Train 3 independent XGBoost regressors (x,y,z) or a multi-output model.
        Here: 3 independent models → clearer interpretation.
        """
        self.models = []
        self.evals_result = {"train": [], "valid": []}

        for dim in range(3):
            print(f"\n📌 Training model for dimension {['X','Y','Z'][dim]}")

            dtrain = xgb.DMatrix(X_train, label=Y_train[:, dim])
            dvalid = xgb.DMatrix(X_valid, label=Y_valid[:, dim])

            evals_res_dim = {}

            model = xgb.train(
                params=self.params,
                dtrain=dtrain,
                num_boost_round=self.num_rounds,
                evals=[(dtrain, "train"), (dvalid, "valid")],
                evals_result=evals_res_dim,
                early_stopping_rounds=self.early_stopping,
                verbose_eval=False
            )

            self.models.append(model)
            self.evals_result[f"train"].append(evals_res_dim["train"]["rmse"])
            self.evals_result[f"valid"].append(evals_res_dim["valid"]["rmse"])

    def predict(self, X):
        preds = []
        dX = xgb.DMatrix(X)
        for model in self.models:
            preds.append(model.predict(dX))
        return np.vstack(preds).T  # shape (N,3)

    def evaluate(self, X_valid, Y_valid):
        pred = self.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(Y_valid, pred))
        return rmse


# ======================================================================
#  EXPERIMENTS
# ======================================================================
class ExperimentRunner:
    """
    Run multiple experiments with different hyperparams.
    """
    def __init__(self, dataset: SatelliteDataset):
        self.dataset = dataset
        self.results = []

    def run_experiment(self, name: str, params: Dict[str, Any], num_rounds=300, early_stopping=20):
        X_train, Y_train = self.dataset.get_train()
        X_valid, Y_valid = self.dataset.get_valid()

        trainer = XGBoostTrainer(params, num_rounds, early_stopping)
        trainer.train(X_train, Y_train, X_valid, Y_valid)
        rmse = trainer.evaluate(X_valid, Y_valid)

        exp_result = {
            "name": name,
            "params": params,
            "rmse": rmse,
            "history": trainer.evals_result
        }
        self.results.append(exp_result)

        print(f"🔍 Experiment '{name}' finished — RMSE = {rmse:.6f}")
        return exp_result

    def plot_results(self):
        plt.figure(figsize=(12, 6))
        for res in self.results:
            # res["history"]["valid"] est une liste de 3 listes (X,Y,Z) de longueurs possiblement différentes
            rmse_lists = res["history"]["valid"]  # shape: 3 x (num_boost_round_dim)
            max_len = max(len(lst) for lst in rmse_lists)
            rmse_padded = np.full((len(rmse_lists), max_len), np.nan, dtype=float)
            for i, lst in enumerate(rmse_lists):
                rmse_padded[i, :len(lst)] = lst
            # moyenne par itération en ignorant les NaN (si early stopping diffère entre dimensions)
            mean_valid_rmse = np.nanmean(rmse_padded, axis=0)
            plt.plot(mean_valid_rmse, label=res["name"])
        plt.title("Comparaison RMSE Validation selon les Epochs (XGBoost)")
        plt.xlabel("Boosting Iteration")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(True)
        plt.show()


# ======================================================================
#  USAGE EXAMPLE
# ======================================================================
if __name__ == "__main__":
    # --------------------------------------------------------------
    # Load real dataset (ISS, SGP4 vs Horizons)
    # --------------------------------------------------------------
    csv_path = "datasetISS_200TLE.csv"  # adapter le chemin si besoin
    df = pd.read_csv(csv_path, sep=";")

    # Supprimer dx, dy, dz si présents
    for col in ["dx_km", "dy_km", "dz_km"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Colonnes d'entrée X (mêmes que pour le GRU)
    X_cols = [
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

    # Conversion des dates en timestamps numériques (secondes)
    if "time_utc" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time_utc"]).astype("int64") / 1e9
    if "tle_epoch" in df.columns:
        df["tle_epoch"] = pd.to_datetime(df["tle_epoch"]).astype("int64") / 1e9

    # Vérification des colonnes Horizons
    def find_col(candidates):
        cols_lower = {c.lower().strip(): c for c in df.columns}
        for cand in candidates:
            key = cand.lower().strip()
            if key in cols_lower:
                return cols_lower[key]
        for key, original in cols_lower.items():
            for cand in candidates:
                if cand.lower().strip() in key:
                    return original
        raise KeyError(f"Aucune colonne trouvée parmi {candidates} dans {df.columns}")

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

    # Optionnel : normalisation simple de X (z-score)
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - X_mean) / X_std

    # Construction du dataset
    dataset = SatelliteDataset(X, Y, train_ratio=0.8)
    runner = ExperimentRunner(dataset)

    # --------------------------------------------------------------
    # EXPERIMENT 1
    params1 = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.001,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    runner.run_experiment("xgb_lr0.001_depth6", params1)

    # --------------------------------------------------------------
    # EXPERIMENT 2
    params2 = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.1,
        "max_depth": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }
    runner.run_experiment("xgb_lr0.1_depth10", params2)

    # --------------------------------------------------------------
    runner.plot_results()
