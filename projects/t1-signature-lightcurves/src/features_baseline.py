"""Hand-crafted variability features: the incumbent baseline.

These are the statistics that survey pipelines have used for two decades, in the lineage
of FATS and `feets`. They are implemented here rather than imported so that the feature
set is fixed, inspectable, and free of the installation problems that affect the published
packages on current Python versions.

The important structural point: almost every feature below is a function of the *marginal
distribution* of magnitudes, or of pairwise differences. Amplitude, skew, kurtosis,
percentile ratios and Stetson K are all invariant to permuting the observations in time.
That is precisely the information a signature keeps and these discard, so this baseline is
not a straw man -- it is the sharpest possible statement of what "order-blind" means.

The handful of features that do see ordering (linear trend, rise and fade timescales,
maximum slope) are included deliberately, because leaving them out would make the
comparison dishonest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["band_features", "object_features", "feature_frame", "FEATURE_NAMES"]

_BAND_FEATURES = [
    "n_obs", "t_span", "mean", "median", "std", "mad", "amplitude", "beyond1std",
    "skew", "kurtosis", "stetson_k", "percent_amplitude", "flux_mid20", "flux_mid50",
    "flux_mid80", "linear_trend", "max_slope", "median_abs_slope", "rise_frac",
    "fade_frac", "peak_mag", "cadence_median", "cadence_iqr",
]


def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def band_features(t: np.ndarray, m: np.ndarray, e: np.ndarray) -> dict[str, float]:
    """Variability features for a single band.

    Magnitudes are used directly (smaller is brighter), so a "peak" is a minimum.
    """
    n = len(t)
    if n < 2:
        return {k: 0.0 for k in _BAND_FEATURES}

    order = np.argsort(t)
    t, m, e = t[order], m[order], e[order]
    span = t[-1] - t[0]

    mean, std = float(np.mean(m)), float(np.std(m, ddof=1)) if n > 1 else 0.0
    median = float(np.median(m))
    mad = float(np.median(np.abs(m - median)))

    p5, p95 = np.percentile(m, [5, 95])
    amplitude = float(p95 - p5)

    beyond1std = float(np.mean(np.abs(m - mean) > std)) if std > 0 else 0.0

    # Stetson K: a kurtosis-like statistic weighted by the reported uncertainties,
    # robust to the heteroscedastic errors typical of survey photometry.
    if n > 1 and np.all(e > 0):
        delta = np.sqrt(n / (n - 1.0)) * (m - mean) / e
        denom = np.sqrt(np.mean(delta ** 2))
        stetson_k = _safe(np.mean(np.abs(delta)) / denom) if denom > 0 else 0.0
    else:
        stetson_k = 0.0

    percent_amplitude = _safe(np.max(np.abs(m - median)) / median) if median != 0 else 0.0

    def _mid(frac: float) -> float:
        lo, hi = (100 - frac) / 2, 100 - (100 - frac) / 2
        a, b = np.percentile(m, [lo, hi])
        return _safe((b - a) / amplitude) if amplitude > 0 else 0.0

    # ordering-sensitive features
    slope = np.polyfit(t - t[0], m, 1)[0] if span > 0 else 0.0
    dt = np.diff(t)
    dm = np.diff(m)
    valid = dt > 0
    slopes = dm[valid] / dt[valid] if valid.any() else np.array([0.0])
    i_peak = int(np.argmin(m))                      # brightest point
    rise_frac = _safe((t[i_peak] - t[0]) / span) if span > 0 else 0.0
    fade_frac = _safe((t[-1] - t[i_peak]) / span) if span > 0 else 0.0

    cad = dt[valid] if valid.any() else np.array([0.0])

    return {
        "n_obs": float(n),
        "t_span": _safe(span),
        "mean": _safe(mean),
        "median": _safe(median),
        "std": _safe(std),
        "mad": _safe(mad),
        "amplitude": amplitude,
        "beyond1std": beyond1std,
        "skew": _safe(pd.Series(m).skew()),
        "kurtosis": _safe(pd.Series(m).kurtosis()),
        "stetson_k": stetson_k,
        "percent_amplitude": percent_amplitude,
        "flux_mid20": _mid(20),
        "flux_mid50": _mid(50),
        "flux_mid80": _mid(80),
        "linear_trend": _safe(slope),
        "max_slope": _safe(np.max(np.abs(slopes))),
        "median_abs_slope": _safe(np.median(np.abs(slopes))),
        "rise_frac": rise_frac,
        "fade_frac": fade_frac,
        "peak_mag": _safe(np.min(m)),
        "cadence_median": _safe(np.median(cad)),
        "cadence_iqr": _safe(np.percentile(cad, 75) - np.percentile(cad, 25)),
    }


def object_features(lc: pd.DataFrame, bands: tuple[str, ...] = ("g", "r")) -> dict[str, float]:
    """Per-band features plus the cross-band colour terms.

    `lc` holds the observations of one object with columns mjd, band, mag, magerr.
    """
    out: dict[str, float] = {}
    per_band: dict[str, dict[str, float]] = {}

    for b in bands:
        sub = lc[lc.band == b]
        feats = band_features(sub.mjd.to_numpy(), sub.mag.to_numpy(), sub.magerr.to_numpy())
        per_band[b] = feats
        out.update({f"{b}_{k}": v for k, v in feats.items()})

    # Colour and cross-band timing. The bands are not sampled simultaneously, so a
    # per-epoch colour does not exist without interpolation; the median difference is
    # the standard interpolation-free substitute.
    if len(bands) >= 2:
        b0, b1 = bands[0], bands[1]
        out[f"colour_{b0}{b1}"] = _safe(per_band[b0]["median"] - per_band[b1]["median"])
        out[f"amp_ratio_{b0}{b1}"] = (
            _safe(per_band[b0]["amplitude"] / per_band[b1]["amplitude"])
            if per_band[b1]["amplitude"] > 0 else 0.0
        )
        out[f"peak_offset_{b0}{b1}"] = _safe(per_band[b0]["rise_frac"] - per_band[b1]["rise_frac"])
    return out


FEATURE_NAMES: list[str] = (
    [f"{b}_{k}" for b in ("g", "r") for k in _BAND_FEATURES]
    + ["colour_gr", "amp_ratio_gr", "peak_offset_gr"]
)


def feature_frame(df: pd.DataFrame, bands: tuple[str, ...] = ("g", "r")) -> pd.DataFrame:
    """Feature table for every object in a long-format photometry frame."""
    rows = []
    for oid, lc in df.groupby("oid", sort=True):
        feats = object_features(lc, bands)
        feats["oid"] = oid
        feats["label"] = lc.label.iloc[0]
        rows.append(feats)
    out = pd.DataFrame(rows).set_index("oid")
    return out
