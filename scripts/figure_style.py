"""
Shared figure style for all paper figures
==========================================

One matplotlib style + one canonical colour per model, imported and applied at
the top of every plotting script so the paper's figures are visually consistent
and legible at A4 two-column width.

Usage
-----
    from figure_style import apply_style, MODEL_COLORS, color_for
    apply_style()
    ...
    ax.plot(x, y, color=color_for("TEMPO Fine-Tuned"))

Design targets (Po point 3, A4 two-column legibility)
-----------------------------------------------------
- axis labels >= 12 pt, ticks >= 10 pt, legend >= 10 pt, titles >= 13 pt,
  in-plot annotations >= 9 pt.
- line width >= 1.8, marker size >= 5.
- one fixed colour per model, reused across EVERY figure (same colour = same
  model everywhere). Palette is Okabe-Ito, colourblind-safe.
- PNG raster output at dpi = 300; PDF/SVG kept as vector (text stays text via
  fonttype 42), tight bounding box.
"""

import matplotlib as mpl

# --- Minimum font sizes (pt). Scripts that hard-code sizes should use these ---
FS_TITLE = 13
FS_SUPTITLE = 14
FS_LABEL = 12
FS_TICK = 10
FS_LEGEND = 10
FS_ANNOT = 9

# --- One canonical colour per model (Okabe-Ito colourblind-safe) -------------
MODEL_COLORS = {
    "Random Forest":    "#0072B2",  # blue
    "XGBoost":          "#E69F00",  # orange
    "LSTM":             "#009E73",  # bluish green
    "TEMPO Zero-Shot":  "#CC79A7",  # reddish purple
    "TEMPO Fine-Tuned": "#D55E00",  # vermillion
    "Persistence":      "#999999",  # grey
}

# Short / alternative keys used by some scripts -> same canonical colour.
_ALIASES = {
    "RF": "Random Forest",
    "XGB": "XGBoost",
    "T-Zero-Shot": "TEMPO Zero-Shot",
    "T-Fine-Tuned": "TEMPO Fine-Tuned",
    "TEMPO ZS": "TEMPO Zero-Shot",
    "TEMPO FT": "TEMPO Fine-Tuned",
}

_FALLBACK = "#607D8B"  # neutral slate for anything unrecognised


def color_for(model):
    """Return the canonical colour for a model name (with alias support)."""
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    if model in _ALIASES:
        return MODEL_COLORS[_ALIASES[model]]
    return _FALLBACK


_RC = {
    # Fonts
    "font.size": FS_LABEL,
    "axes.titlesize": FS_TITLE,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "figure.titlesize": FS_SUPTITLE,
    # Lines / markers
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "axes.linewidth": 0.8,
    # Grid
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    # Saving: PNG raster crisp; PDF/SVG stay vector with real text
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.dpi": 110,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def apply_style():
    """Apply the shared rcParams. Call once at the top of a plotting script."""
    mpl.rcParams.update(_RC)


# Convenience: the model draw order used across bar/line figures.
MODEL_ORDER = [
    "Random Forest", "XGBoost", "LSTM",
    "TEMPO Zero-Shot", "TEMPO Fine-Tuned",
]
