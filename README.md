<div align="center">

# 🛰️ Satellite

### Learning to Correct SGP4 Orbit Propagation Error with Deep Sequence Models

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-MLP%20%7C%20LSTM%20%7C%20GRU%20%7C%20Transformer-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="pandas" src="https://img.shields.io/badge/pandas-time--series-150458?logo=pandas&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/status-academic%20research-orange">
  <img alt="License" src="https://img.shields.io/badge/license-unspecified-lightgrey">
</p>

<p><i>Applied deep learning for astrodynamics — orbit propagation error modeling</i></p>

[Overview](#-overview) • [Problem](#-problem-statement) • [Data](#-data) • [Models](#-models) • [Results](#-results) • [Usage](#-installation--usage) • [Structure](#-repository-structure)

</div>

---

## 📌 Overview

Satellite orbits are commonly predicted from **Two-Line Element sets (TLEs)** using the **SGP4** analytical propagator — fast, but only approximate, since it ignores or simplifies many perturbation forces (drag variability, third-body effects, solar radiation pressure...). Its position error grows over time as the propagated state drifts away from the TLE epoch.

This project asks a simple question: **can a neural network learn to predict SGP4's own error**, using only orbital-element features available at inference time, so that the propagated position can be corrected?

To find out, four sequence-aware architectures (**MLP, LSTM, GRU, Transformer**) are trained to regress the residual vector between an SGP4-propagated position and a high-precision **JPL Horizons** ephemeris, on real orbital data for two satellites (**ISS** and **HST**).

<p align="center">
  <img src="suite/plots/comparaisons erreurs .png" alt="Model error comparison" width="80%">
  <br><sub><b>Fig.</b> Comparative error curves across model architectures (see <code>suite/plots/</code> for the full figure set).</sub>
</p>

---

## 🎯 Problem Statement

Given the orbital state of a satellite at time *t* — derived from its most recent TLE and the SGP4-propagated position — predict the **position error vector**:

$$\vec{e}(t) = \vec{r}_{\text{Horizons}}(t) - \vec{r}_{\text{SGP4}}(t) = (err_x,\ err_y,\ err_z) \in \mathbb{R}^3 \text{ (km)}$$

where `r_Horizons` is treated as ground truth (JPL Horizons ephemeris) and `r_SGP4` is the fast analytical propagation. Learning to predict `e(t)` from cheap, TLE-derived features would let a lightweight correction be added on top of SGP4 without paying for a full numerical propagator.

This is framed as a **multivariate time-series regression** problem: each model consumes a sliding window of past orbital-state feature vectors and predicts the 3-D error at the window's last timestep.

---

## 🗂️ Data

Two real datasets are used, each combining SGP4 propagation with JPL Horizons ephemeris for a specific satellite:

| Dataset | Satellite | Rows | Description |
|---|---|---:|---|
| `datasetISS_200TLE.csv` | ISS (ZARYA) | ~17.6k | Positions sampled at fixed time steps against 200 successive TLEs |
| `dataset_hst_sgp4_vs_horizons2.csv` | Hubble Space Telescope | ~48.2k | Same schema, longer/denser time coverage |

### Feature schema (`;`-separated CSV)

| Column | Meaning |
|---|---|
| `time_utc`, `tle_epoch` | Sample timestamp and epoch of the TLE used (parsed to UNIX time) |
| `dt_since_tle_s` | Elapsed time since the TLE epoch — proxy for propagation-error growth |
| `mean_motion`, `mean_motion_derivative` | Orbital revolutions/day and its drift |
| `orbital_speed_km_s`, `altitude_drift_km_per_day` | Derived kinematic quantities |
| `bstar` | Drag term from the TLE |
| `inclination_deg`, `raan_deg`, `eccentricity`, `arg_perigee_deg`, `mean_anomaly_deg` | Classical Keplerian elements |
| `rev_number` | Revolution count since launch |
| `x_sgp4_km`, `y_sgp4_km`, `z_sgp4_km` | SGP4-propagated ECI position (model input) |
| `x_horizons_km`, `y_horizons_km`, `z_horizons_km` | JPL Horizons reference position (used only to build the target) |

**Target** — computed on the fly in every training script, then the raw Horizons/`dx,dy,dz` columns are dropped to prevent leakage:

```python
err_x = x_horizons_km - x_sgp4_km
err_y = y_horizons_km - y_sgp4_km
err_z = z_horizons_km - z_sgp4_km
```

All 17 input features are `StandardScaler`-normalized (`(X - mean) / std`) before windowing.

---

## 🧠 Models

Every architecture is trained on **sliding windows** of consecutive samples and predicts the 3-D error vector at the last timestep of the window.

| Model | Script | Sequence length | Architecture | Notes |
|---|---|:---:|---|---|
| **MLP** | `MLP_Approach_2.py` | 128 | 3 FC layers `(256‑256‑128)` + ReLU + Dropout, uses only the **last** timestep of the window | Non-recurrent baseline |
| **LSTM** | `LSTM_Approach_2.py` | 50 | `nn.LSTM` (1–3 layers, 64–128 hidden units) → FC head | Early stopping on validation loss |
| **GRU** | `GRU_Approach_2.py` | 128 | `nn.GRU` (1–3 layers, 64–128 hidden units, dropout) → FC head | Lighter recurrent alternative to LSTM |
| **Transformer** | `Transformer_Approach_2.py` | — | Linear embedding + `LayerNorm` → `TransformerEncoder` (`d_model=128`, `nhead=4`, 3–4 layers) → FC head | AdamW + LR warmup, early stopping |

All models are trained with **MSE loss** and evaluated with **MSE/RMSE** on chronological **train / validation / test** splits (≈70 / 15 / 15%, taken as contiguous, non-shuffled slices to respect the time-series nature of the data). Each script sweeps a small grid of hyperparameters (learning rate, hidden size, depth) as independent experiments, logging one metrics/predictions CSV pair per configuration under `results/`.

> Device selection is automatic (`mps` → `cuda` → `cpu`), so the same scripts run unmodified on Apple Silicon, CUDA GPUs, or CPU-only machines.

---

## 📊 Results

> ⚠️ **Not all numbers are directly comparable.** `MLP` / `LSTM` / `GRU` are trained on the **ISS** dataset, while the committed `Transformer` results come from the **HST** dataset — different orbital regime, different data scale. Each block below only compares runs on the same dataset.

### ISS dataset — MLP vs. LSTM (test RMSE, km)

| Model | Configuration | Test RMSE |
|---|---|:---:|
| MLP | `lr=1e-4`, hidden `(256,256,128)` | **2.37** |
| MLP | `lr=1e-4`, hidden `(512,256,128)` | 2.44 |
| MLP | `lr=5e-4`, hidden `(256,256,128)` | 3.21 |
| MLP | `lr=1e-3`, hidden `(256,256,128)` | 3.67 |
| LSTM | `hidden=128`, `layers=1` | **2.20** |
| LSTM | `hidden=64`, `layers=3` | 2.20 |
| LSTM | `hidden=64`, `layers=1` | 2.22 |
| LSTM | `hidden=128`, `layers=3` | 2.28 |

- Lower learning rates generalize noticeably better for the MLP — the higher-LR runs reach a lower **train** error but a visibly worse **test** error, a classic overfitting signature.
- LSTM configurations cluster tightly (2.20 – 2.28 km) regardless of depth/width, suggesting the recurrent inductive bias helps but further capacity brings diminishing returns on this dataset size.
- ⚠️ **Validation/test asymmetry**: across all ISS runs, the **validation RMSE is far higher (~17–21 km) than the test RMSE (~2.2–3.7 km)**. Since splits are contiguous chronological slices, this points to the validation segment covering a different orbital/TLE regime than train and test — worth investigating (e.g. stratified or rolling-window splits) before trusting these numbers as a generalization estimate.

### HST dataset — Transformer

| Model | Test RMSE |
|---|:---:|
| Transformer | 0.43 (HST scale — not comparable to the ISS table above) |

### GRU

The committed GRU results (`faux/results/Metrics_GRU_*.csv`) come from an earlier iteration with a different target normalization and are not on the same scale as the tables above — treat them as exploratory rather than final numbers. Re-running `GRU_Approach_2.py` regenerates comparable, up-to-date results under `results/`.

All raw metrics/prediction CSVs and diagnostic plots are available under [`results/`](./results) and [`suite/plots/`](./suite/plots).

---

## 🗃️ Repository Structure

```text
Satellite/
├── MLP_Approach_2.py                  # MLP baseline — current pipeline
├── LSTM_Approach_2.py                 # LSTM — current pipeline
├── GRU_Approach_2.py                  # GRU — current pipeline
├── Transformer_Approach_2.py          # Transformer — current pipeline (HST dataset)
├── datasetISS_200TLE.csv              # ISS: SGP4 vs. Horizons, 200 TLEs
├── dataset_hst_sgp4_vs_horizons2.csv  # HST: SGP4 vs. Horizons
├── results/                           # Metrics + test predictions, current pipeline
├── suite/                             # Later analysis pass: extra plots & result snapshots
│   ├── plots/                         # Learning curves, error comparisons, overfitting checks
│   └── results/
├── erreur/results/                    # Earlier experiment snapshot (MLP/LSTM error CSVs)
├── faux/results/                      # Earlier experiment snapshot (GRU/LSTM/MLP, different normalization)
└── Autre/                             # Archive: exploratory scripts, notebooks, duplicate variants,
                                        # a background-reading PDF, and saved model checkpoints
```

> The repository currently mixes a **cleaned "Approach 2" pipeline** (top-level `*_Approach_2.py` scripts) with several earlier iterations (`erreur/`, `faux/`, `suite/`, `Autre/`). If you're only interested in the current, reproducible pipeline, the top-level scripts and `results/` are the ones to use.

---

## 🚀 Installation & Usage

```bash
git clone https://github.com/cyyyp100/Satellite.git
cd Satellite

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install torch numpy pandas
```

> No `requirements.txt` is committed — the list above covers the actual imports used by the training scripts (`os`, `time` are standard library).

Each model is a standalone script that runs its own hyperparameter sweep and writes results to `results/`:

```bash
python MLP_Approach_2.py           # ISS dataset — MLP sweep (4 configs)
python LSTM_Approach_2.py          # ISS dataset — LSTM sweep (6 configs)
python GRU_Approach_2.py           # ISS dataset — GRU sweep (3 configs)
python Transformer_Approach_2.py   # HST dataset — Transformer
```

Each run prints per-epoch train/valid/test MSE & RMSE, then saves, per configuration:

- `results/Metrics_<Model>_<config>.csv` — full training history (loss/RMSE per epoch)
- `results/Test_Predictions_<Model>_<config>.csv` — SGP4 position, Horizons position, and predicted error on the test split

---

## 🔭 Limitations & Future Work

- **Validation/test discrepancy** on the ISS runs (see [Results](#-results)) suggests the contiguous chronological split may not be representative — worth revisiting with a rolling-origin or stratified-by-TLE split.
- **Cross-dataset comparability**: Transformer results are on HST, not ISS — a fair head-to-head across all four architectures on the same satellite/dataset is still missing.
- Repository hygiene: no `.gitignore` (tracked `__pycache__/`, `.DS_Store`), no `requirements.txt`, and several overlapping result folders (`erreur/`, `faux/`, `suite/`, `Autre/`) from earlier iterations that would benefit from being consolidated or archived under a single `experiments/` or `archive/` directory.
- A natural extension would be evaluating whether the predicted correction, added back to the raw SGP4 output, actually reduces propagation error at operationally relevant horizons (e.g. multi-day forecasts), rather than only minimizing RMSE on the residual itself.

---

## 📝 License

No license is currently declared in this repository. Since this is an academic/research project, adding an explicit license (e.g. MIT, or CC BY-NC 4.0 for non-commercial use) is recommended before any external distribution or reuse.

---

<div align="center">
<sub>Generated from the training scripts, datasets, and result files committed to this repository.</sub>
</div>
