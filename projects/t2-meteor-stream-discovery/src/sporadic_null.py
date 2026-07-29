"""Three models of the sporadic background, and the reason the first one is not enough.

The consensus sweep produced 471 unmatched structures against 64 recovered known showers.
A method whose novel-detection rate exceeds its known-recovery rate sevenfold is measuring
its own false positives, and the leading suspect is the null.

WHY PERMUTATION FAILS. Shuffling each orbital element independently preserves every
marginal distribution and destroys every correlation between them. But the sporadic complex
is not marginal structure -- the helion, antihelion, apex and toroidal sources exist
*because* orbital elements are correlated in particular ways. A permutation null therefore
models a background that does not exist, and every real correlation in the data, stream or
not, registers against it as an excess.

Three nulls are implemented so the choice can be measured rather than argued:

  permutation   The original. Marginals preserved, all joint structure destroyed.
                Retained as the reference that the others must beat.

  kde           A kernel-density model of the joint distribution, sampled. Preserves
                correlations at scales larger than the bandwidth and erases them below it.
                The bandwidth is therefore the whole design: it must sit ABOVE the scale of
                a meteoroid stream (D_SH ~ 0.05-0.1, a few degrees in the angles) and BELOW
                the scale of the sporadic sources (tens of degrees). Chosen physically and
                stated, not tuned to produce a desired answer -- cross-validated bandwidth
                would fit the streams too and defeat the purpose.

  sideband      The physical control, and parameter-free where it counts. A stream is
                active over a bounded range of solar longitude, so the background either
                side of that range -- offset by a gap, at the same ecliptic geometry -- is
                the same sporadic complex with that stream absent. If the KDE null is
                right, it should agree with this one; if they disagree, the KDE bandwidth
                is wrong.

Angles are handled on the circle. Node and argument of perihelion are periodic, so they
enter the density estimate as (cos, sin) pairs rather than as raw degrees, where 359 and 1
would otherwise sit at opposite ends of the range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

__all__ = ["permutation_null", "kde_null", "sideband_null", "NULL_MODELS",
           "PHYSICAL_SCALES"]

Q_EARTH, A_EARTH = 0.9833, 1.0167
COLS = ("q", "e", "i", "node", "peri")

# Physical scales used to whiten each dimension before a single-bandwidth KDE. Each is
# chosen to be comfortably LARGER than the internal spread of a meteoroid stream and
# smaller than the extent of a sporadic source, so the estimate smooths streams away while
# keeping the complex.
PHYSICAL_SCALES = {
    "q": 0.10,        # AU; stream dispersion in q is a few 0.01 AU
    "e": 0.08,
    "i": 6.0,         # degrees
    "node_xy": 0.12,  # on the unit circle, ~7 degrees
    "peri_xy": 0.12,
}
KDE_BANDWIDTH = 1.0  # in whitened units, i.e. exactly the scales above


def earth_crossing(q: np.ndarray, e: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = np.where(e < 1.0, q * (1.0 + e) / np.maximum(1.0 - e, 1e-12), np.inf)
    return (q <= A_EARTH) & (Q >= Q_EARTH)


def _valid(df: pd.DataFrame) -> pd.DataFrame:
    ok = (earth_crossing(df.q.to_numpy(), df.e.to_numpy())
          & (df.q > 0) & (df.e >= 0) & (df.e < 1.2)
          & (df.i >= 0) & (df.i <= 180))
    return df[ok]


# --------------------------------------------------------------------- permutation

def permutation_null(df: pd.DataFrame, rng: np.random.Generator, size: int,
                     max_tries: int = 12) -> pd.DataFrame:
    """Element-wise shuffle restricted to Earth-crossing orbits (the original null)."""
    kept, have = [], 0
    for _ in range(max_tries):
        draw = pd.DataFrame({c: rng.permutation(df[c].to_numpy()) for c in COLS})
        ok = _valid(draw)
        if len(ok):
            kept.append(ok)
            have += len(ok)
        if have >= size:
            break
    if not kept:
        return df[list(COLS)].sample(min(size, len(df)), replace=True,
                                     random_state=int(rng.integers(1 << 30)))
    out = pd.concat(kept, ignore_index=True)
    return out.iloc[:size].reset_index(drop=True)


# --------------------------------------------------------------------- KDE

def _to_whitened(df: pd.DataFrame) -> np.ndarray:
    node = np.radians(df.node.to_numpy())
    peri = np.radians(df.peri.to_numpy())
    return np.column_stack([
        df.q.to_numpy() / PHYSICAL_SCALES["q"],
        df.e.to_numpy() / PHYSICAL_SCALES["e"],
        df.i.to_numpy() / PHYSICAL_SCALES["i"],
        np.cos(node) / PHYSICAL_SCALES["node_xy"],
        np.sin(node) / PHYSICAL_SCALES["node_xy"],
        np.cos(peri) / PHYSICAL_SCALES["peri_xy"],
        np.sin(peri) / PHYSICAL_SCALES["peri_xy"],
    ])


def _from_whitened(x: np.ndarray) -> pd.DataFrame:
    node = np.degrees(np.arctan2(x[:, 4] * PHYSICAL_SCALES["node_xy"],
                                 x[:, 3] * PHYSICAL_SCALES["node_xy"])) % 360.0
    peri = np.degrees(np.arctan2(x[:, 6] * PHYSICAL_SCALES["peri_xy"],
                                 x[:, 5] * PHYSICAL_SCALES["peri_xy"])) % 360.0
    return pd.DataFrame({
        "q": x[:, 0] * PHYSICAL_SCALES["q"],
        "e": x[:, 1] * PHYSICAL_SCALES["e"],
        "i": np.clip(x[:, 2] * PHYSICAL_SCALES["i"], 0.0, 180.0),
        "node": node, "peri": peri,
    })


def kde_null(df: pd.DataFrame, rng: np.random.Generator, size: int,
             bandwidth: float = KDE_BANDWIDTH, max_tries: int = 8) -> pd.DataFrame:
    """Sample a kernel-density model of the joint orbital-element distribution.

    Correlations larger than the bandwidth survive -- which is what keeps the sporadic
    sources in the null -- while structure below it is smoothed away, which is what removes
    the streams.
    """
    fit = df[list(COLS)].dropna()
    if len(fit) < 50:
        return permutation_null(df, rng, size)
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(_to_whitened(fit))

    kept, have = [], 0
    for _ in range(max_tries):
        # KernelDensity.sample takes its own seed; draw one from our generator so the
        # whole pipeline stays reproducible from a single root seed.
        drawn = kde.sample(n_samples=size, random_state=int(rng.integers(1 << 30)))
        ok = _valid(_from_whitened(drawn))
        if len(ok):
            kept.append(ok)
            have += len(ok)
        if have >= size:
            break
    if not kept:
        return permutation_null(df, rng, size)
    return pd.concat(kept, ignore_index=True).iloc[:size].reset_index(drop=True)


# --------------------------------------------------------------------- sideband

def sideband_null(archive: pd.DataFrame, sol_lon_centre: float, rng: np.random.Generator,
                  size: int, gap_deg: float = 8.0, width_deg: float = 10.0,
                  shift_node: bool = True) -> pd.DataFrame:
    """Real meteors from solar longitudes either side of the window, offset by a gap.

    The physical control: actual observed orbits carrying the full correlation structure of
    the sporadic complex, sampled where the window's own streams are not active.

    THE NODE SHIFT IS NOT OPTIONAL. A first version omitted it and the null collapsed to
    zero density everywhere -- it predicted no meteors at sporadic regions and none at the
    Perseids either, which looks like perfect discrimination and is actually total failure.
    The reason is geometric: a meteoroid can only be observed where its orbit crosses
    Earth's, so its ascending node is locked to the solar longitude of the encounter. Orbits
    borrowed from 18 degrees away therefore carry nodes 18 degrees away, which puts every
    one of them outside any D_SH ball centred in the window. Rotating the node to the
    window's own solar longitude restores the comparison and is the physically correct
    operation: it asks what the same sporadic population would look like had Earth met it
    here.
    """
    lo1 = (sol_lon_centre - gap_deg - width_deg) % 360.0
    hi1 = (sol_lon_centre - gap_deg) % 360.0
    lo2 = (sol_lon_centre + gap_deg) % 360.0
    hi2 = (sol_lon_centre + gap_deg + width_deg) % 360.0

    def band(lo, hi):
        return ((archive.sol_lon >= lo) & (archive.sol_lon < hi)) if lo <= hi else \
               ((archive.sol_lon >= lo) | (archive.sol_lon < hi))

    pool = archive[band(lo1, hi1) | band(lo2, hi2)]
    if len(pool) < 50:
        return permutation_null(archive, rng, size)
    take = pool.sample(min(size, len(pool)), replace=len(pool) < size,
                       random_state=int(rng.integers(1 << 30)))
    out = take[list(COLS)].reset_index(drop=True)
    if shift_node:
        # Rotate each borrowed orbit's node by the solar-longitude offset it was drawn
        # from, so it sits at the encounter geometry of the target window.
        offset = (sol_lon_centre - take.sol_lon.to_numpy()) % 360.0
        out["node"] = (out.node.to_numpy() + offset) % 360.0
    return out


NULL_MODELS = {"permutation": "marginals only, all correlations destroyed",
               "kde": "joint density, smoothed below the stream scale",
               "sideband": "real orbits from adjacent solar longitudes"}
