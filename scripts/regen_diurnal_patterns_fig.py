"""Regenerate figures/diurnal_patterns_by_ecosystem.{pdf,png} (paper Fig 4).

Faithful reproduction of the mean-diurnal-cycle figure in
notebooks/01_data_exploration.ipynb (cell 10): group by hour-of-day, plot the
mean NEE per hour. Reads only committed raw per-site series
(data/raw/<n>.<site>.*, column NEE_VUT_REF). No model inference; no number
changed.

Two deliberate changes vs the notebook (both approved):
  1. Ecosystem regrouping. The notebook had a separate "Cropland Sites" panel
     for IE-Cra and FR-LGt, which are actually wetlands (raised bog and fen;
     data/metadata/site_characteristics.csv). We collapse to TWO panels --
     "Wetland Sites" (FI-Lom, GL-ZaF, IE-Cra, DE-Akm, FR-LGt, UK-AMo) and
     "Forest Sites" (SE-Htm) -- which is the correct grouping and reinforces the
     wetland-vs-forest amplitude contrast.
  2. Shared y-axis across both panels, so the forest's large-amplitude diurnal
     cycle is directly comparable to the suppressed, low-amplitude wetlands.

Fonts enlarged for A4 two-column legibility via scripts/figure_style.py. Site
traces use a distinct qualitative palette (the model-colour convention applies
to models, not sites).
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from figure_style import apply_style, FS_TITLE, FS_LABEL, FS_LEGEND

apply_style()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "figures"
PAPER_FIG_DIR = ROOT / "paper" / "figures"   # the dir the paper compiles from

# (file, ecosystem, split). IE-Cra / FR-LGt corrected Cropland -> Wetland.
sites = {
    "FI-Lom": ("1.FI-Lom.csv",  "Wetland", "Training"),
    "GL-ZaF": ("2.GL-ZaF.csv",  "Wetland", "Training"),
    "IE-Cra": ("3.IE-Cra.xlsx", "Wetland", "Training"),
    "DE-Akm": ("4.DE-Akm.csv",  "Wetland", "Training"),
    "FR-LGt": ("5.FR-LGt.csv",  "Wetland", "Training"),
    "UK-AMo": ("6.UK-AMo.csv",  "Wetland", "Test"),
    "SE-Htm": ("7.SE-Htm.csv",  "Forest",  "Test"),
}

# Distinct qualitative colours for site traces (Okabe-Ito-derived, CB-safe).
SITE_COLORS = {
    "FI-Lom": "#0072B2", "GL-ZaF": "#E69F00", "IE-Cra": "#009E73",
    "DE-Akm": "#CC79A7", "FR-LGt": "#56B4E9", "UK-AMo": "#D55E00",
    "SE-Htm": "#0072B2",
}


def hourly_means():
    out = {}
    for code, (fname, eco, split) in sites.items():
        fp = DATA_DIR / fname
        df = pd.read_excel(fp) if fp.suffix == ".xlsx" else pd.read_csv(fp)
        ts = "TIMESTAMP" if "TIMESTAMP" in df.columns else "timestamp"
        df[ts] = pd.to_datetime(df[ts], dayfirst=True)
        df = df.set_index(ts)
        s = df[["NEE_VUT_REF"]].copy()
        s["hour"] = s.index.hour
        out[code] = (s.groupby("hour")["NEE_VUT_REF"].mean(), eco, split)
    return out


def main():
    hm = hourly_means()
    groups = {"Wetland Sites": [c for c in sites if sites[c][1] == "Wetland"],
              "Forest Sites":  [c for c in sites if sites[c][1] == "Forest"]}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for idx, (title, codes) in enumerate(groups.items()):
        ax = axes[idx]
        for code in codes:
            mean_by_hour, _eco, split = hm[code]
            style = "--" if split == "Test" else "-"
            ax.plot(mean_by_hour.index, mean_by_hour.values,
                    marker="o", linestyle=style, label=code,
                    linewidth=2.0, markersize=5, color=SITE_COLORS[code])
        ax.axhline(y=0, color="red", linestyle="--", linewidth=1.2)
        ax.axvspan(6, 18, alpha=0.08, color="gold")
        ax.set_xlabel("Hour of Day", fontsize=FS_LABEL)
        if idx == 0:
            ax.set_ylabel("Mean NEE (μmol CO₂ m⁻² s⁻¹)", fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=FS_LEGEND, ncol=2 if len(codes) > 3 else 1)
        ax.set_xlim(0, 23)

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "diurnal_patterns_by_ecosystem.png")
    fig.savefig(FIG_DIR / "diurnal_patterns_by_ecosystem.pdf")
    plt.close(fig)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        shutil.copy(FIG_DIR / f"diurnal_patterns_by_ecosystem.{ext}",
                    PAPER_FIG_DIR / f"diurnal_patterns_by_ecosystem.{ext}")
    print("Saved: figures/ + paper/figures/ diurnal_patterns_by_ecosystem.{png,pdf} (2 panels)")


if __name__ == "__main__":
    main()
