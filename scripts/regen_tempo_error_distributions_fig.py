"""Regenerate figures/tempo_error_distributions.{pdf,png} (paper Fig 3).

Faithful reproduction of the error-distribution figure in
notebooks/03_tempo_analysis.ipynb (cell 20). Reads only committed saved arrays
-- test targets (data/processed/test_<site>_y.npy) and model predictions
(results/predictions/ and results/predictions/baselines/) -- so NO model
inference runs and no number changes. UK-AMo relabelled "Peatland" -> "Wetland"
for consistency with the paper. Fonts enlarged via scripts/figure_style.py.
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_style import (apply_style, color_for, FS_TITLE, FS_SUPTITLE,
                          FS_LABEL, FS_LEGEND)

apply_style()

ROOT = Path(__file__).resolve().parent.parent
PREDS_DIR = ROOT / "results" / "predictions"
DATA_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"
PAPER_FIG_DIR = ROOT / "paper" / "figures"   # the dir the paper compiles from

SITES = ["UK-AMo", "SE-Htm"]
SITE_INFO = {
    "UK-AMo": {"ecosystem": "Wetland"},   # corrected from "Peatland"
    "SE-Htm": {"ecosystem": "Forest"},
}
# display name -> (parent dir, file key). Overlay = the three the notebook shows.
MODEL_FILES = {
    "XGBoost":          (PREDS_DIR / "baselines", "xgboost_preds"),
    "TEMPO Zero-Shot":  (PREDS_DIR,               "tempo_zero_shot_preds"),
    "TEMPO Fine-Tuned": (PREDS_DIR,               "tempo_fine_tuned_preds"),
}
OVERLAY = ["XGBoost", "TEMPO Zero-Shot", "TEMPO Fine-Tuned"]


def load():
    targets = {s: np.load(DATA_DIR / f"test_{s}_y.npy") for s in SITES}
    preds = {}
    for name, (parent, key) in MODEL_FILES.items():
        preds[name] = {}
        for s in SITES:
            p = parent / f"{key}_{s}.npy"
            if p.exists():
                arr = np.load(p)
                if not np.isnan(arr).any():
                    preds[name][s] = arr
    return targets, preds


def main():
    targets, preds = load()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax_idx, site in enumerate(SITES):
        ax = axes[ax_idx]
        actual_flat = targets[site].flatten()
        for model in OVERLAY:
            if site not in preds.get(model, {}):
                continue
            errors = preds[model][site].flatten() - actual_flat
            ax.hist(errors, bins=80, density=True, alpha=0.45,
                    color=color_for(model), label=model, edgecolor="none")
        ax.axvline(0, color="black", ls="--", lw=1.5)
        ax.set_xlabel("Prediction Error", fontsize=FS_LABEL)
        ax.set_ylabel("Density", fontsize=FS_LABEL)
        ax.set_title(f"{site} ({SITE_INFO[site]['ecosystem']})",
                     fontsize=FS_TITLE, fontweight="bold")
        ax.legend(fontsize=FS_LEGEND)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Prediction Error Distributions",
                 fontsize=FS_SUPTITLE, fontweight="bold")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "tempo_error_distributions.png")
    fig.savefig(FIG_DIR / "tempo_error_distributions.pdf")
    plt.close(fig)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        shutil.copy(FIG_DIR / f"tempo_error_distributions.{ext}",
                    PAPER_FIG_DIR / f"tempo_error_distributions.{ext}")
    print("Saved: figures/ + paper/figures/ tempo_error_distributions.{png,pdf}")


if __name__ == "__main__":
    main()
