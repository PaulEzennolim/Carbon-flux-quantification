"""
Univariate-baseline control experiment (reviewer comment 1: covariate confound)
==============================================================================

Question
--------
Is the baselines' wetland->forest degradation caused by BIOME MISMATCH, or by
their 19 exogenous covariates becoming noise out of domain? To separate the two,
we give the baselines EXACTLY the information TEMPO gets -- the NEE history and
nothing else -- and re-run the identical benchmark.

Why the inputs must be rebuilt (verified in Step 0)
---------------------------------------------------
data/processed/*_X.npy are (N, 336, 19) and contain 19 EXOGENOUS covariates in
this order:
    SW_IN_F, LW_IN_F, VPD_F, TA_F, PA_F, P_F, WS_F, G_F_MDS, LE_F_MDS, H_F_MDS,
    MODIS_band_1..7, DOY, TOD
NEE_VUT_REF is NOT among them -- it is the TARGET only (y). TEMPO, by contrast,
consumes the NEE lookback window itself (models/tempo_carbon_flux.py:
load_nee_series -> create_sequences -> [B, 336, 1]). So the univariate baselines
cannot be a subset of the existing covariates; the NEE lookback (N, 336, 1) has
to be constructed from the raw series, exactly as TEMPO does.

Design (everything except the input information is held fixed)
--------------------------------------------------------------
- Windows      : 336-h lookback -> 96-h horizon, built from data/raw NEE_VUT_REF
                 with ffill/bfill -- byte-identical construction to TEMPO's.
                 Verified: the resulting y matches data/processed/*_y.npy exactly
                 (train and both test sites), so splits/targets are unchanged.
- Train pool   : the 5 wetland sites, concatenated in the same order.
- Test sites   : UK-AMo, SE-Htm (held out, untouched).
- Scaling      : mirrors the existing pipeline's convention -- a StandardScaler
                 fit on the TRAINING inputs only, applied to train and test; the
                 TARGET is left in raw NEE units (as in tempo_data_prep.py, which
                 scales the 19 covariates with a train-fit global scaler and
                 leaves the target raw). Scaling NEE is a deterministic affine
                 map of the same series, so it adds no information: the models
                 still see only NEE history.
                 NOTE: the pipeline does NOT use per-site z-scoring; we follow the
                 existing convention so the ONLY thing that changes vs the
                 multivariate run is the input information (19 covariates -> NEE).
- Models/HPs   : the real training code (models/baseline_models.py,
                 models/lstm_baseline.py, scripts/train_baselines.py) with the
                 SAME hyperparameters (RF/XGB 200 estimators; 2-layer LSTM,
                 128 units) and seed 42. Nothing is tuned. The LSTM's input_size
                 necessarily changes 19 -> 1, so its parameter count differs; we
                 report it rather than forcing it.
- Metrics      : the SAME evaluators as the existing benchmark (sklearn r2_score
                 for RF/XGB via baseline_models.evaluate; pooled R^2 for the LSTM
                 via lstm_baseline.evaluate_lstm), so numbers are directly
                 comparable to all_models_summary.csv.

Output
------
results/metrics/univariate_baselines_summary.csv   (NEW file; nothing existing
is overwritten). Existing predictions/metrics are left untouched.
"""

import sys
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# The real training code (module-level sets SEED=42, np/torch seeds).
import train_baselines as tb  # noqa: E402
import torch  # noqa: E402

LOOKBACK, HORIZON = 336, 96
TRAIN_SITES = ["FI-Lom", "GL-ZaF", "IE-Cra", "DE-Akm", "FR-LGt"]
TEST_SITES = ["UK-AMo", "SE-Htm"]
SITE_FILES = {
    "FI-Lom": "1.FI-Lom.csv", "GL-ZaF": "2.GL-ZaF.csv", "IE-Cra": "3.IE-Cra.xlsx",
    "DE-Akm": "4.DE-Akm.csv", "FR-LGt": "5.FR-LGt.csv",
    "UK-AMo": "6.UK-AMo.csv", "SE-Htm": "7.SE-Htm.csv",
}
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
OUT_CSV = ROOT / "results" / "metrics" / "univariate_baselines_summary.csv"


def load_nee(site):
    """Raw NEE_VUT_REF with ffill/bfill -- identical to TEMPO's load_nee_series."""
    p = RAW_DIR / SITE_FILES[site]
    df = pd.read_excel(p) if p.suffix == ".xlsx" else pd.read_csv(p)
    return df["NEE_VUT_REF"].ffill().bfill().values.astype(np.float64)


def make_sequences(series):
    """336-h lookback -> 96-h horizon sliding windows (TEMPO's create_sequences)."""
    n = len(series) - LOOKBACK - HORIZON + 1
    X = np.stack([series[i:i + LOOKBACK] for i in range(n)])
    y = np.stack([series[i + LOOKBACK:i + LOOKBACK + HORIZON] for i in range(n)])
    return X, y


def build():
    print("[1/4] Building univariate NEE inputs (336 -> 96) from data/raw ...")
    Xs, ys = [], []
    for s in TRAIN_SITES:
        X, y = make_sequences(load_nee(s))
        Xs.append(X); ys.append(y)
        print(f"      {s:8} {len(X):>6,} sequences")
    X_train = np.concatenate(Xs)
    y_train = np.concatenate(ys)

    test = {}
    for s in TEST_SITES:
        X, y = make_sequences(load_nee(s))
        test[s] = {"X": X, "y": y}
        print(f"      {s:8} {len(X):>6,} sequences (held out)")

    # --- Verify alignment with the existing benchmark's splits/targets ---
    assert np.allclose(y_train, np.load(PROC_DIR / "train_y.npy")), "train y mismatch"
    for s in TEST_SITES:
        assert np.allclose(test[s]["y"], np.load(PROC_DIR / f"test_{s}_y.npy")), f"{s} y mismatch"
    print("      ✓ targets identical to data/processed/*_y.npy (splits unchanged)")

    # --- Scale inputs with a TRAIN-fit scaler; target stays in raw NEE units ---
    scaler = StandardScaler().fit(X_train.reshape(-1, 1))
    def scale(A):
        return scaler.transform(A.reshape(-1, 1)).reshape(A.shape)[..., None]  # (N,336,1)
    X_train_s = scale(X_train)
    for s in TEST_SITES:
        test[s]["X"] = scale(test[s]["X"])
    print(f"      ✓ input scaled (train-fit StandardScaler); X_train={X_train_s.shape}")
    return X_train_s, y_train, test


def penalty(r2_wet, r2_for):
    """Existing benchmark formula: (R2_wetland - R2_forest) / R2_wetland."""
    return (r2_wet - r2_for) / r2_wet


def main():
    print("=" * 74)
    print("UNIVARIATE BASELINES — NEE history only (TEMPO's information)")
    print("=" * 74)
    X_train, y_train, test = build()

    results = {}

    print("\n[2/4] Random Forest (200 trees, depth 15, seed 42) ...")
    rf_res, _ = tb.train_random_forest(X_train, y_train, test)
    results["Random Forest"] = rf_res

    print("\n[3/4] XGBoost (200 estimators, depth 8, lr 0.01) ...")
    xgb_res, _ = tb.train_xgboost(X_train, y_train, test)
    results["XGBoost"] = xgb_res

    print("\n[4/4] LSTM (2 layers x 128 units, input_size=1) ...")
    n_params = sum(p.numel() for p in
                   tb.LSTMForecaster(input_size=1, hidden_size=128,
                                     num_layers=2, horizon=HORIZON).parameters())
    print(f"      LSTM parameter count (input_size=1): {n_params:,}")
    lstm_res, _ = tb.train_lstm_model(X_train, y_train, test)
    results["LSTM"] = lstm_res

    # --- Write NEW csv (never touches all_models_summary.csv) ---
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Site", "RMSE", "MAE", "R2", "Inputs"])
        for model, per_site in results.items():
            for site in TEST_SITES:
                m = per_site[site]
                w.writerow([model, site, f"{m['RMSE']:.4f}", f"{m['MAE']:.4f}",
                            f"{m['R2']:.4f}", "NEE history only (336x1)"])
    print(f"\nSaved: {OUT_CSV.relative_to(ROOT)}")

    print("\n" + "=" * 74)
    print("UNIVARIATE RESULTS (NEE history only)")
    print("=" * 74)
    print(f"{'Model':<16}{'Site':<10}{'RMSE':>9}{'MAE':>9}{'R2':>9}")
    for model, per_site in results.items():
        for site in TEST_SITES:
            m = per_site[site]
            print(f"{model:<16}{site:<10}{m['RMSE']:>9.4f}{m['MAE']:>9.4f}{m['R2']:>9.4f}")

    print("\nTransfer penalty  (R2_wetland - R2_forest) / R2_wetland")
    print(f"{'Model':<16}{'R2 UK-AMo':>11}{'R2 SE-Htm':>11}{'penalty':>10}")
    for model, per_site in results.items():
        w_, f_ = per_site["UK-AMo"]["R2"], per_site["SE-Htm"]["R2"]
        print(f"{model:<16}{w_:>11.4f}{f_:>11.4f}{penalty(w_, f_)*100:>9.1f}%")
    print(f"\nLSTM parameter count (input_size=1): {n_params:,}")
    print("=" * 74)


if __name__ == "__main__":
    main()
