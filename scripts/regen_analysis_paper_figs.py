"""Regenerate the three paper figures owned by the big analysis scripts, using
their REAL plotting functions on ALREADY-SAVED results (no model retraining, so
no number changes), with the shared figure style applied:

  * transfer_matrix_heatmap.png   <- transfer_learning_analysis.plot_transfer_matrix
                                     on results/transfer_learning/transfer_matrix.csv
  * model_comparison_by_horizon.png <- horizon_analysis.plot_model_comparison
                                     on saved predictions
  * error_by_hour_{UK-AMo,SE-Htm}.{pdf,png} <- error_analysis.plot_error_by_hour
                                     on saved predictions

Each output is written to its canonical figures/<subdir>/ location and copied to
paper/figures/ (the directory the paper compiles from).
"""
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from figure_style import apply_style

PAPER_FIG = ROOT / "paper" / "figures"
PAPER_FIG.mkdir(parents=True, exist_ok=True)


def _copy(src: Path):
    shutil.copy(src, PAPER_FIG / src.name)
    print(f"  copied -> paper/figures/{src.name}")


def do_transfer_matrix():
    print("[1/3] transfer_matrix_heatmap.png (from saved transfer_matrix.csv)")
    import transfer_learning_analysis as tl
    apply_style()
    csv = ROOT / "results" / "transfer_learning" / "transfer_matrix.csv"
    df = pd.read_csv(csv)
    fig = tl.plot_transfer_matrix(df)
    out = tl.FIG_DIR / "transfer_matrix_heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.relative_to(ROOT)}")
    _copy(out)


def do_horizon():
    print("[2/3] model_comparison_by_horizon.png (from saved predictions)")
    import horizon_analysis as hz
    apply_style()
    all_metrics = {}
    for site in hz.SITES:
        data = hz.load_site_data(site)
        y_true = data["y_true"]
        all_metrics[site] = {
            model: hz.horizon_metrics(y_true, y_pred)
            for model, y_pred in data["preds"].items()
        }
    fig = hz.plot_model_comparison(all_metrics)
    out = hz.FIG_DIR / "model_comparison_by_horizon.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.relative_to(ROOT)}")
    _copy(out)


def do_error_by_hour():
    print("[3/3] error_by_hour_{UK-AMo,SE-Htm}.{pdf,png} (from saved predictions)")
    import error_analysis as ea
    apply_style()
    # Two models per site: fine-tuned TEMPO + that site's strongest baseline
    # (per all_models_summary.csv R2): Random Forest is best on UK-AMo (wetland),
    # XGBoost is best on SE-Htm (forest). TEMPO panel first, baseline second.
    KEEP = {
        "UK-AMo": ["TEMPO Fine-Tuned", "Random Forest"],
        "SE-Htm": ["TEMPO Fine-Tuned", "XGBoost"],
    }
    for site in ["UK-AMo", "SE-Htm"]:
        data = ea.load_site_data(site)
        # Restrict to the two chosen models (ordered) without touching the
        # plotting code -- plot_error_by_hour renders data['predictions'] as-is.
        preds = data["predictions"]
        data["predictions"] = {m: preds[m] for m in KEEP[site] if m in preds}
        got = list(data["predictions"].keys())
        assert got == KEEP[site], f"{site}: expected {KEEP[site]}, got {got}"
        print(f"  {site}: panels = {got}")
        ea.plot_error_by_hour(data, site)   # writes .png and .pdf to ea.FIG_DIR
        for ext in ("pdf", "png"):
            _copy(ea.FIG_DIR / f"error_by_hour_{site}.{ext}")


if __name__ == "__main__":
    do_transfer_matrix()
    do_horizon()
    do_error_by_hour()
    print("\nDone: 3 analysis-owned paper figures regenerated + copied.")
