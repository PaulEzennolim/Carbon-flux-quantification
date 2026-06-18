"""Regenerate figures/all_models_comparison.{png,pdf} with corrected site label.

Faithful standalone reproduction of the figure cell in
notebooks/03_tempo_analysis.ipynb. It reads only the committed summary metrics
(results/metrics/all_models_summary.csv) -- no data/processed targets or model
inference are required -- and relabels UK-AMo as "Wetland" (was "Peatland").
Styling (model order, colours, layout, TEMPO bar highlight) matches the
notebook. Output is written to figures/ and copied into paper/figures/.
"""
from pathlib import Path
import shutil
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "metrics" / "all_models_summary.csv"
FIG_DIR = ROOT / "figures"
PAPER_FIG_DIR = ROOT / "paper" / "figures"

SITES = ["UK-AMo", "SE-Htm"]
SITE_INFO = {
    "UK-AMo": {"ecosystem": "Wetland"},   # corrected from "Peatland"
    "SE-Htm": {"ecosystem": "Forest"},
}
MODEL_COLORS = {
    "Random Forest":    "#3498db",
    "XGBoost":          "#2ecc71",
    "LSTM":             "#e74c3c",
    "TEMPO Zero-Shot":  "#f39c12",
    "TEMPO Fine-Tuned": "#9b59b6",
}
MODEL_ORDER = list(MODEL_COLORS.keys())

df = pd.read_csv(CSV)                       # columns: Model,Site,RMSE,MAE,R2
df = df.rename(columns={"R2": "R²"})   # match notebook metric label
metrics_list = ["RMSE", "MAE", "R²"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for row_idx, site in enumerate(SITES):
    site_df = df[df["Site"] == site].copy()
    site_df["Model"] = pd.Categorical(site_df["Model"], categories=MODEL_ORDER, ordered=True)
    site_df = site_df.sort_values("Model").reset_index(drop=True)
    for col_idx, metric in enumerate(metrics_list):
        ax = axes[row_idx, col_idx]
        models = site_df["Model"].values
        values = site_df[metric].values
        colors = [MODEL_COLORS.get(m, "gray") for m in models]
        bars = ax.bar(range(len(models)), values, color=colors, alpha=0.85)
        for i, m in enumerate(models):
            if "TEMPO" in str(m):
                bars[i].set_edgecolor("black")
                bars[i].set_linewidth(2)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        short = [str(m).replace("TEMPO ", "T-").replace("Random Forest", "RF") for m in models]
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(metric, fontsize=11, fontweight="bold")
        ax.set_title(f"{site} ({SITE_INFO[site]['ecosystem']})", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

plt.suptitle("Cross-Site Model Performance Comparison", fontsize=15, fontweight="bold", y=1.0)
plt.tight_layout()
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIG_DIR / "all_models_comparison.png", dpi=300, bbox_inches="tight")
plt.savefig(FIG_DIR / "all_models_comparison.pdf", bbox_inches="tight")

PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy(FIG_DIR / "all_models_comparison.pdf", PAPER_FIG_DIR / "all_models_comparison.pdf")
print("Regenerated all_models_comparison.{png,pdf} with UK-AMo labelled 'Wetland'.")
print("Copied PDF -> paper/figures/all_models_comparison.pdf")
