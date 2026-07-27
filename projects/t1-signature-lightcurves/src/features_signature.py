"""Turn a multi-band light curve into a path, and the path into signature features.

The awkward part of applying signatures to survey photometry is not the signature; it is
deciding what the path *is*. Observations in different filters are taken at different
instants, so there is no single moment at which a two-band vector exists. Any construction
of a joint multi-channel path therefore involves a choice, and different choices are not
equivalent. Three are implemented, and the ablation in `run_benchmark.py` measures which
one matters rather than assuming.

  per_band     Each filter becomes its own two-dimensional path (time, magnitude), and the
               signatures are concatenated. No cross-band information at all, and no
               interpolation of any kind. The conservative choice.

  interleaved  All observations, in time order, form one path with channels
               (time, magnitude, band indicator). Cross-band ordering is preserved
               exactly, but the magnitude channel jumps between filters with different
               zero points, so the increments mix a real signal with a filter offset.

  joint_ffill  A common time axis with the last observed magnitude carried forward in each
               band, giving (time, mag_g, mag_r). This is piecewise-constant
               interpolation. It is included precisely because it is the compromise most
               practitioners would reach for, and calling it interpolation-free would be
               dishonest -- it is interpolation, of the cheapest kind.

Scaling: the k-th signature level scales as the k-th power of the path, so raw
coefficients span many orders of magnitude across levels. A signed log-modulus transform,
sign(x) log(1+|x|), compresses that range without discarding sign or ordering, and is
applied before any classifier sees the features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signature import add_basepoint, lead_lag, signature, signature_dim

__all__ = ["build_path", "signature_features", "signature_feature_frame",
           "NORMALISATIONS"]

PATH_MODES = ("per_band", "interleaved", "joint_ffill")


NORMALISATIONS = ("unit_time", "raw_time", "raw")


def _normalise(t: np.ndarray, m: np.ndarray, how: str = "unit_time"
               ) -> tuple[np.ndarray, np.ndarray]:
    """Prepare the (time, magnitude) channels before the path is built.

    The choice matters more than it looks, and the first version of this module got it
    wrong. Scaling time to [0, 1] discards the duration of the event, and centring
    magnitudes discards the absolute brightness -- and for transients both are strong
    physical discriminants that the hand-crafted baseline keeps (as `t_span` and
    `peak_mag`). Throwing them away and then reporting that signatures lose would be
    measuring the preprocessing, not the representation.

      unit_time  time scaled to [0, 1], magnitudes centred on the median. Pure shape.
      raw_time   time in days from the first detection, magnitudes centred. Keeps
                 duration, discards distance-driven brightness.
      raw        time in days, magnitudes uncentred. Keeps everything.
    """
    if how not in NORMALISATIONS:
        raise ValueError(f"normalisation must be one of {NORMALISATIONS}, got {how!r}")
    span = t.max() - t.min()
    if how == "unit_time":
        tn = (t - t.min()) / span if span > 0 else np.zeros_like(t)
        return tn, m - np.median(m)
    tn = t - t.min()
    return (tn, m - np.median(m)) if how == "raw_time" else (tn, m)


def build_path(lc: pd.DataFrame, mode: str = "per_band",
               bands: tuple[str, ...] = ("g", "r"),
               with_time: bool = True, norm: str = "unit_time") -> list[np.ndarray]:
    """Construct one or more paths from a single object's observations.

    Returns a list of paths, since `per_band` yields one per filter and the other modes
    yield a single joint path.
    """
    if mode not in PATH_MODES:
        raise ValueError(f"mode must be one of {PATH_MODES}, got {mode!r}")

    lc = lc.sort_values("mjd")

    if mode == "per_band":
        paths = []
        for b in bands:
            sub = lc[lc.band == b]
            if len(sub) < 2:
                paths.append(np.zeros((2, 2)))
                continue
            tn, mn = _normalise(sub.mjd.to_numpy(), sub.mag.to_numpy(), norm)
            paths.append(np.column_stack([tn, mn]) if with_time else mn.reshape(-1, 1))
        return paths

    if mode == "interleaved":
        keep = lc[lc.band.isin(bands)]
        if len(keep) < 2:
            return [np.zeros((2, 3))]
        tn, mn = _normalise(keep.mjd.to_numpy(), keep.mag.to_numpy(), norm)
        indicator = np.array([bands.index(b) for b in keep.band], dtype=float)
        cols = [tn, mn, indicator] if with_time else [mn, indicator]
        return [np.column_stack(cols)]

    # joint_ffill: piecewise-constant carry-forward onto the union of observation times
    keep = lc[lc.band.isin(bands)]
    if len(keep) < 2:
        return [np.zeros((2, 1 + len(bands)))]
    times = np.sort(keep.mjd.unique())
    cols = []
    for b in bands:
        sub = keep[keep.band == b].sort_values("mjd")
        if len(sub) == 0:
            cols.append(np.zeros_like(times))
            continue
        vals = sub.mag.to_numpy()
        if norm != "raw":
            vals = vals - np.median(vals)
        idx = np.searchsorted(sub.mjd.to_numpy(), times, side="right") - 1
        idx = np.clip(idx, 0, len(sub) - 1)
        cols.append(vals[idx])
    span = times.max() - times.min()
    tn = ((times - times.min()) / span if span > 0 else np.zeros_like(times)) \
        if norm == "unit_time" else times - times.min()
    stacked = [tn] + cols if with_time else cols
    return [np.column_stack(stacked)]


def _log_modulus(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def signature_features(lc: pd.DataFrame, depth: int = 3, mode: str = "per_band",
                       bands: tuple[str, ...] = ("g", "r"), with_time: bool = True,
                       use_lead_lag: bool = False, norm: str = "unit_time",
                       basepoint: bool = False) -> np.ndarray:
    """Signature feature vector for one object.

    `basepoint` matters more than it sounds. The signature depends only on increments, so
    two light curves differing by a constant magnitude offset have identical signatures --
    which is why the `raw` and `raw_time` preparations score identically, a fact worth
    treating as a check on the implementation rather than a coincidence. Prepending the
    origin restores absolute brightness, and for transients that is a real discriminant
    the hand-crafted baseline already has as `peak_mag`.
    """
    paths = build_path(lc, mode=mode, bands=bands, with_time=with_time, norm=norm)
    feats = []
    for p in paths:
        if basepoint:
            p = add_basepoint(p)
        if use_lead_lag:
            p = lead_lag(p)
        feats.append(signature(p, depth))
    return _log_modulus(np.concatenate(feats))


def _expected_dim(depth: int, mode: str, bands: tuple[str, ...], with_time: bool,
                  use_lead_lag: bool) -> int:
    if mode == "per_band":
        d = 2 if with_time else 1
        d = 2 * d if use_lead_lag else d
        return len(bands) * signature_dim(d, depth)
    if mode == "interleaved":
        d = 3 if with_time else 2
    else:
        d = 1 + len(bands) if with_time else len(bands)
    d = 2 * d if use_lead_lag else d
    return signature_dim(d, depth)


def signature_feature_frame(df: pd.DataFrame, depth: int = 3, mode: str = "per_band",
                            bands: tuple[str, ...] = ("g", "r"), with_time: bool = True,
                            use_lead_lag: bool = False, norm: str = "unit_time",
                            basepoint: bool = False) -> pd.DataFrame:
    """Signature features for every object in a long-format photometry frame."""
    dim = _expected_dim(depth, mode, bands, with_time, use_lead_lag)
    rows, index, labels = [], [], []
    for oid, lc in df.groupby("oid", sort=True):
        v = signature_features(lc, depth=depth, mode=mode, bands=bands,
                               with_time=with_time, use_lead_lag=use_lead_lag, norm=norm,
                               basepoint=basepoint)
        if v.size != dim:  # guard against a silently ragged feature matrix
            raise ValueError(f"{oid}: got {v.size} features, expected {dim}")
        rows.append(v)
        index.append(oid)
        labels.append(lc.label.iloc[0])
    out = pd.DataFrame(np.vstack(rows), index=pd.Index(index, name="oid"),
                       columns=[f"sig{i}" for i in range(dim)])
    out["label"] = labels
    return out
