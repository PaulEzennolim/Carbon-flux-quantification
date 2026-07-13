"""Regenerate figures/nee_timeseries_all_sites.{pdf,png} (paper Fig 1).

Faithful standalone reproduction of the figure cell in
notebooks/01_data_exploration.ipynb (cell 8). Reads only the committed raw
per-site series (data/raw/<n>.<site>.*, column NEE_VUT_REF) -- no model
inference. Two changes vs the notebook:
  * ecosystem labels for IE-Cra and FR-LGt corrected "Cropland" -> "Wetland"
    (authoritative: data/metadata/site_characteristics.csv; a raised bog and a
    fen, both wetlands);
  * fonts enlarged for A4 two-column legibility via scripts/figure_style.py.
Per-panel Std/Mean/Range annotations are computed from the SAME raw series each
panel plots (unchanged from the notebook), so UK-AMo Std stays ~3.59.
No underlying numbers are changed.
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import pandas as pd

from figure_style import apply_style, FS_TITLE, FS_LABEL, FS_ANNOT

apply_style()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "figures"
PAPER_FIG_DIR = ROOT / "paper" / "figures"   # the dir the paper compiles from

# (file, name, country, ecosystem, split) -- ecosystem for IE-Cra/FR-LGt
# corrected to Wetland (was "Cropland" in the notebook). Site names for IE-Cra
# and FR-LGt corrected to the ICOS-registry names (Clara Bog / La Guette;
# metadata CSV is authoritative, the notebook "Crogen"/"Laignes" are stale).
sites = {
    "FI-Lom": ("1.FI-Lom.csv",  "Lompolojankka",    "Finland",  "Wetland", "Training"),
    "GL-ZaF": ("2.GL-ZaF.csv",  "Zackenberg Fen",   "Greenland","Wetland", "Training"),
    "IE-Cra": ("3.IE-Cra.xlsx", "Clara Bog",        "Ireland",  "Wetland", "Training"),
    "DE-Akm": ("4.DE-Akm.csv",  "Anklam",           "Germany",  "Wetland", "Training"),
    "FR-LGt": ("5.FR-LGt.csv",  "La Guette",        "France",   "Wetland", "Training"),
    "UK-AMo": ("6.UK-AMo.csv",  "Auchencorth Moss", "UK",       "Wetland", "Test"),
    "SE-Htm": ("7.SE-Htm.csv",  "Hyltemossa",       "Sweden",   "Forest",  "Test"),
}


def load():
    data = {}
    for code, (fname, name, country, eco, split) in sites.items():
        fp = DATA_DIR / fname
        df = pd.read_excel(fp) if fp.suffix == ".xlsx" else pd.read_csv(fp)
        ts = "TIMESTAMP" if "TIMESTAMP" in df.columns else "timestamp"
        df[ts] = pd.to_datetime(df[ts], dayfirst=True)
        df = df.rename(columns={ts: "TIMESTAMP"}).set_index("TIMESTAMP")
        data[code] = {"df": df, "name": name, "ecosystem": eco, "split": split}
    return data


# Half-page two-column layout (Po's sketch): left column = the five training
# sites, right column = the two test sites. A 10-row x 2-col gridspec gives each
# training panel 2 rows and each test panel 5 rows, so both columns fill the same
# height and the two test panels align with the five training panels.
TRAIN = ["FI-Lom", "GL-ZaF", "IE-Cra", "DE-Akm", "FR-LGt"]
TEST = ["UK-AMo", "SE-Htm"]


def _draw_panel(ax, code, info, is_bottom):
    nee = info["df"]["NEE_VUT_REF"]
    # Thin trace: raw hourly series has thousands of points; a thick line
    # would smear detail. (The >=1.8 lw rule targets model-result lines.)
    ax.plot(info["df"].index, nee, linewidth=0.6, alpha=0.7, color="darkblue")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.2, alpha=0.6)

    # Split (Training/Test) is encoded by title colour (green/red); the suffix
    # is dropped for compactness. Left-aligned so the title never reaches the
    # date axis's year-offset (bottom-right) of the panel above it.
    split_color = "red" if info["split"] == "Test" else "green"
    ax.set_title(
        f"{code}: {info['name']} ({info['ecosystem']})",
        fontsize=FS_TITLE, fontweight="bold", color=split_color, pad=4, loc="left",
    )
    # Short y-label (one word, single line) avoids the vertical text collision
    # between stacked panels; the unit is stated once in the figure caption.
    ax.set_ylabel("NEE", fontsize=FS_LABEL)
    ax.grid(True, alpha=0.3)

    # Compact date ticks: each site spans a different period, so panels can't
    # share an x-axis. ConciseDateFormatter keeps the labels short so they don't
    # collide with the title of the panel below in the half-height layout.
    loc = mdates.AutoDateLocator(maxticks=5)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.tick_params(axis="x", labelsize=8)

    stats_text = (
        f"Mean: {nee.mean():.2f}\n"
        f"Std: {nee.std():.2f}\n"
        f"Range: [{nee.min():.1f}, {nee.max():.1f}]"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment="top", fontsize=FS_ANNOT,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    if is_bottom:
        ax.set_xlabel("Date", fontsize=FS_LABEL)


def main():
    data = load()
    # ~half A4 text height when rendered at \textwidth (~7 in): figsize aspect
    # 12:8 -> rendered height 7 * 8/12 ~ 4.7 in.
    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(10, 2, figure=fig, hspace=2.1, wspace=0.20)

    for i, code in enumerate(TRAIN):                       # left column, 2 rows each
        ax = fig.add_subplot(gs[2 * i:2 * i + 2, 0])
        _draw_panel(ax, code, data[code], is_bottom=(i == len(TRAIN) - 1))

    test_spans = [gs[0:5, 1], gs[5:10, 1]]                 # right column, 5 rows each
    for i, code in enumerate(TEST):
        ax = fig.add_subplot(test_spans[i])
        _draw_panel(ax, code, data[code], is_bottom=(i == len(TEST) - 1))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "nee_timeseries_all_sites.png")
    fig.savefig(FIG_DIR / "nee_timeseries_all_sites.pdf")
    plt.close(fig)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        shutil.copy(FIG_DIR / f"nee_timeseries_all_sites.{ext}",
                    PAPER_FIG_DIR / f"nee_timeseries_all_sites.{ext}")
    print("Saved: figures/ + paper/figures/ nee_timeseries_all_sites.{png,pdf}")


if __name__ == "__main__":
    main()
