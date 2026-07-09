"""
Signal-Regularity Analysis of Per-Site NEE Series
==================================================

For each of the 7 EC-tower sites this script quantifies how *regular*
(diurnally repeatable) the target NEE signal is, using three
complementary measures computed directly from that site's hourly
NEE_VUT_REF series -- the exact target that feeds the models in
`tempo_data_prep.py` (`target = df['NEE_VUT_REF']`).

Measures (per site)
-------------------
1. spectral_entropy  : Shannon entropy of the normalised power spectrum,
                       divided by log(N_freq) so it lies in [0, 1].
                       LOWER  = spectral power concentrated in few
                       frequencies = MORE regular / predictable.
2. power_24h         : Fraction of total spectral power falling in the
                       frequency bin closest to the 24-hour period.
                       HIGHER = stronger diurnal cycle.
3. autocorr_24h      : Autocorrelation of the (mean-removed) series at a
                       lag of 24 hours. HIGHER = today's hour looks like
                       the same hour yesterday = more diurnally regular.

Amplitude (magnitude-preserving) measures added per site
--------------------------------------------------------
4. abs_power_24h     : ABSOLUTE 24-h spectral power = power_24h (fraction)
                       x total signal variance. Unlike power_24h this keeps
                       magnitude, so a strong-but-noisy site is not equated
                       with a strong-and-clean one.
5. diurnal_amplitude : Peak-to-trough range of the MEAN diurnal cycle:
                       group by hour-of-day, take mean NEE per hour, then
                       max - min. Computed on the filled series; also
                       cross-checked on the raw (pre-fill) series.
6. nee_std           : Standard deviation of the site's NEE series (a plain
                       magnitude sanity reference).
7. nan_fraction_before_fill : gap-fill caveat carried alongside the data.

Method / choices (documented and printed at runtime)
----------------------------------------------------
- Source           : data/raw/<file>, column NEE_VUT_REF (target used by
                     the models). Timestamp column parsed with
                     dayfirst=True (matches tempo_data_prep.py).
- Sampling         : hourly. Series is reindexed onto a *complete* hourly
                     grid so the spectrum is evenly sampled (required by
                     Welch / FFT). Inserted-timestamp count is reported.
- NaN handling     : explicit. Raw NaNs (plus any grid gaps) are linearly
                     interpolated internally, then edge NaNs are
                     forward/backward filled. The pre-fill NaN fraction is
                     reported per site. (The model target itself is filled
                     with ffill/bfill; interpolation is used here because
                     flat ffill runs would artificially inflate low-freq
                     power and lag-24 autocorrelation.)
- Spectrum         : Welch periodogram (scipy.signal.welch), fs = 1.0
                     cycle/hour, Hann window, 50% overlap, detrend =
                     'linear' (removes slow trends so the diurnal peak is
                     not swamped). nperseg chosen as the largest multiple
                     of 24 that fits (<= 672 h = 4 weeks) so a frequency
                     bin lands exactly on the 24-h period.
- Autocorrelation  : statsmodels.tsa.stattools.acf (falls back to a numpy
                     implementation if statsmodels is unavailable), lag 24,
                     on the mean-removed filled series.

Output
------
results/analysis/signal_regularity.csv with columns:
    site, ecosystem, spectral_entropy, power_24h, autocorr_24h, n_hours
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

try:
    from statsmodels.tsa.stattools import acf as _sm_acf
    _HAVE_SM = True
except Exception:  # pragma: no cover - fallback path
    _HAVE_SM = False


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_CSV = ROOT / "results" / "analysis" / "signal_regularity.csv"
META_CSV = ROOT / "data" / "metadata" / "site_characteristics.csv"

TARGET_COL = "NEE_VUT_REF"

SITE_FILES = {
    "FI-Lom": "1.FI-Lom.csv",
    "GL-ZaF": "2.GL-ZaF.csv",
    "IE-Cra": "3.IE-Cra.xlsx",  # Excel
    "DE-Akm": "4.DE-Akm.csv",
    "FR-LGt": "5.FR-LGt.csv",
    "UK-AMo": "6.UK-AMo.csv",
    "SE-Htm": "7.SE-Htm.csv",
}

# Analysis constants
FS = 1.0                 # samples per hour
PERIOD_24H = 24.0        # hours
TARGET_FREQ = 1.0 / PERIOD_24H  # cycles per hour
MAX_NPERSEG = 24 * 28    # 4 weeks; keep it a multiple of 24 for an exact bin
LAG_24 = 24


def load_nee_series(site):
    """Load a site's hourly NEE_VUT_REF onto a complete hourly grid.

    Returns (series_on_grid, n_raw, n_nan_raw, n_inserted) where
    series_on_grid still contains NaNs (filling happens later).
    """
    path = RAW_DIR / SITE_FILES[site]
    df = pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_csv(path)

    ts_col = "TIMESTAMP" if "TIMESTAMP" in df.columns else "timestamp"
    df[ts_col] = pd.to_datetime(df[ts_col], dayfirst=True)
    df = df.set_index(ts_col).sort_index()

    s = df[TARGET_COL].astype(float)
    # Collapse any exact-duplicate timestamps
    s = s[~s.index.duplicated(keep="first")]

    n_raw = len(s)
    n_nan_raw = int(s.isna().sum())

    # Reindex onto a regular, gap-free hourly grid
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="1h")
    n_inserted = len(full_idx) - len(s.index)
    s = s.reindex(full_idx)

    return s, n_raw, n_nan_raw, max(n_inserted, 0)


def fill_series(s):
    """Explicit NaN handling: linear interpolation + edge ffill/bfill.

    Returns (filled_series, nan_fraction_before_fill). The filled series
    keeps its DatetimeIndex so hour-of-day grouping stays available.
    """
    nan_frac = float(s.isna().mean())
    filled = s.interpolate(method="linear", limit_direction="both")
    filled = filled.ffill().bfill()
    return filled, nan_frac


def diurnal_amplitude(s):
    """Peak-to-trough range of the mean diurnal cycle.

    Groups by hour-of-day, averages NEE per hour (NaN-aware), then returns
    max - min. Works on either the filled or the raw series; with the raw
    series, hours with no valid data are skipped via nanmean.
    """
    by_hour = s.groupby(s.index.hour).mean()
    if by_hour.notna().sum() < 2:
        return np.nan
    return float(by_hour.max() - by_hour.min())


def spectral_measures(x):
    """Return (normalised_spectral_entropy, normalised_power_at_24h)."""
    nperseg = int(min(MAX_NPERSEG, (len(x) // 24) * 24))
    if nperseg < 24:  # not enough data for even one diurnal cycle window
        return np.nan, np.nan

    freqs, psd = welch(
        x,
        fs=FS,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="linear",
        scaling="density",
    )

    # Drop the DC (zero-frequency) component; regularity is about the
    # oscillatory structure, and DC would otherwise dominate entropy/power.
    mask = freqs > 0
    freqs, psd = freqs[mask], psd[mask]

    total = psd.sum()
    if total <= 0:
        return np.nan, np.nan

    p = psd / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    norm_entropy = float(entropy / np.log(len(psd)))

    # Power in the bin closest to the 24-h frequency, as a fraction of total
    idx24 = int(np.argmin(np.abs(freqs - TARGET_FREQ)))
    power_24h = float(psd[idx24] / total)

    return norm_entropy, power_24h


def autocorr_lag(x, lag):
    """Lag-`lag` autocorrelation of the mean-removed series."""
    if _HAVE_SM:
        vals = _sm_acf(x, nlags=lag, fft=True, missing="raise")
        return float(vals[lag])
    # numpy fallback (Pearson autocorrelation at the given lag)
    xc = x - x.mean()
    denom = np.dot(xc, xc)
    if denom == 0:
        return np.nan
    return float(np.dot(xc[:-lag], xc[lag:]) / denom)


def load_ecosystems():
    meta = pd.read_csv(META_CSV)
    return dict(zip(meta["Site_Code"], meta["Ecosystem_Type"]))


def main():
    print("=" * 72)
    print("SIGNAL-REGULARITY ANALYSIS  --  per-site NEE_VUT_REF")
    print("=" * 72)
    print(f"Source dir      : {RAW_DIR}")
    print(f"Target column   : {TARGET_COL}")
    print(f"Sampling freq   : {FS} sample/hour (hourly)")
    print(f"Welch window    : Hann, nperseg<= {MAX_NPERSEG} h (multiple of 24),"
          f" 50% overlap, detrend='linear'")
    print(f"24-h frequency  : {TARGET_FREQ:.6f} cycles/hour")
    print(f"Autocorr lag    : {LAG_24} h   (statsmodels acf: {_HAVE_SM})")
    print("-" * 72)

    ecosystems = load_ecosystems()
    rows = []

    for site in SITE_FILES:
        s_raw, n_raw, n_nan_raw, n_inserted = load_nee_series(site)
        s_filled, nan_frac = fill_series(s_raw)
        x = s_filled.to_numpy(dtype=float)
        n_hours = int(len(x))

        ent, pow24 = spectral_measures(x)
        ac24 = autocorr_lag(x, LAG_24)

        # Amplitude / magnitude-preserving measures
        nee_std = float(np.std(x))
        variance = float(np.var(x))
        abs_pow24 = pow24 * variance if pd.notna(pow24) else np.nan
        diur_amp = diurnal_amplitude(s_filled)
        diur_amp_raw = diurnal_amplitude(s_raw)  # cross-check on unfilled data

        eco = ecosystems.get(site, "Unknown")
        rows.append({
            "site": site,
            "ecosystem": eco,
            "spectral_entropy": ent,
            "power_24h": pow24,
            "abs_power_24h": abs_pow24,
            "diurnal_amplitude": diur_amp,
            "nee_std": nee_std,
            "autocorr_24h": ac24,
            "nan_fraction_before_fill": nan_frac,
            "n_hours": n_hours,
        })

        amp_delta = (diur_amp - diur_amp_raw) if pd.notna(diur_amp_raw) else np.nan
        amp_note = (f"raw={diur_amp_raw:6.3f} (Δfilled={amp_delta:+.3f})"
                    if pd.notna(diur_amp_raw) else "raw=NaN")
        print(f"[{site}] eco={eco:<8} n_raw={n_raw:>6} "
              f"grid_hours={n_hours:>6} inserted={n_inserted:>5} "
              f"NaN_before_fill={nan_frac*100:5.1f}%  "
              f"diurnal_amp filled={diur_amp:6.3f} {amp_note}")

    df = pd.DataFrame(rows, columns=[
        "site", "ecosystem", "spectral_entropy", "power_24h",
        "abs_power_24h", "diurnal_amplitude", "nee_std",
        "autocorr_24h", "nan_fraction_before_fill", "n_hours",
    ])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("-" * 72)
    print(f"Saved: {OUT_CSV.relative_to(ROOT)}")
    print("=" * 72)

    # Pretty table
    show = df.copy()
    for c in ["spectral_entropy", "power_24h", "abs_power_24h",
              "diurnal_amplitude", "nee_std", "autocorr_24h",
              "nan_fraction_before_fill"]:
        show[c] = show[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "NaN")
    print(show.to_string(index=False))
    print("=" * 72)

    # ── KEY DIAGNOSTIC: do the measures order the 2 R2-known sites like R2? ──
    # Only UK-AMo and SE-Htm have a zero-shot TEMPO R2. R2: SE-Htm(0.752) >
    # UK-AMo(0.384), i.e. forest > wetland. For each measure we report whether
    # SE-Htm scores higher or lower than UK-AMo, and whether that ordering
    # matches R2. (No correlation claim -- only 2 R2 points.)
    R2 = {"UK-AMo": 0.384, "SE-Htm": 0.752}
    di = df.set_index("site")

    # For each measure: does a HIGHER value mean "more regular / stronger"?
    # R2 order is SE-Htm > UK-AMo. A measure "agrees" if it orders the two the
    # same way R2 does, given the measure's own direction of regularity.
    #   spectral_entropy: LOWER = more regular  -> agrees if SE-Htm < UK-AMo
    #   all others:       HIGHER = stronger      -> agrees if SE-Htm > UK-AMo
    measures = [
        ("spectral_entropy", "lower"),
        ("power_24h", "higher"),
        ("abs_power_24h", "higher"),
        ("diurnal_amplitude", "higher"),
        ("nee_std", "higher"),
        ("autocorr_24h", "higher"),
    ]

    print("KEY DIAGNOSTIC -- 2 sites with zero-shot TEMPO R2 (forest vs wetland)")
    print("-" * 72)
    header = (f"{'site':<8}{'eco':<9}{'entropy':>9}{'pow24_f':>9}"
              f"{'abs_p24':>10}{'diur_amp':>10}{'nee_std':>9}{'R2':>8}")
    print(header)
    for site in ["SE-Htm", "UK-AMo"]:
        r = di.loc[site]
        print(f"{site:<8}{r['ecosystem']:<9}{r['spectral_entropy']:>9.4f}"
              f"{r['power_24h']:>9.4f}{r['abs_power_24h']:>10.4f}"
              f"{r['diurnal_amplitude']:>10.4f}{r['nee_std']:>9.4f}{R2[site]:>8.3f}")
    print("-" * 72)
    print(f"R2 ordering: SE-Htm ({R2['SE-Htm']}) > UK-AMo ({R2['UK-AMo']})"
          f"  =>  forest > wetland")
    print("Per-measure ordering vs R2:")
    for m, direction in measures:
        se, uk = float(di.loc["SE-Htm", m]), float(di.loc["UK-AMo", m])
        se_higher = se > uk
        # "more regular" for SE-Htm?
        if direction == "lower":
            se_more_regular = se < uk
        else:
            se_more_regular = se > uk
        verdict = "SAME as R2 (agrees)" if se_more_regular else "OPPOSITE to R2"
        rel = "higher" if se_higher else "lower"
        print(f"  {m:<20} SE-Htm={se:>10.4f}  UK-AMo={uk:>10.4f}  "
              f"SE-Htm is {rel:<6} ({direction}=more regular)  -> {verdict}")
    print("=" * 72)


if __name__ == "__main__":
    main()
