"""Isolate the mechanism: can each representation see the ORDER of events?

Before asking whether signatures help on real photometry, it is worth establishing that
the claimed mechanism exists at all, in a setting where the answer is known by
construction.

Two classes of synthetic light curve are built as double-peaked events that differ *only*
in which peak comes first:

  class A   a bright peak followed by a faint one
  class B   a faint peak followed by a bright one

The two peaks sit symmetrically about the midpoint of the observing window and the curve
starts and ends at the same baseline, so class B is the exact time reverse of class A.
Three consequences follow by construction rather than by measurement:

  1. The marginal distribution of magnitudes is the same in both classes, so every
     statistic of that distribution -- mean, median, amplitude, skew, kurtosis, percentile
     ratios, Stetson K -- is uninformative.
  2. The net change from first to last observation is zero in both classes, so the
     *first* signature level, which is exactly that net increment, is uninformative too.
  3. What differs is the order of the two excursions, which is what iterated integrals
     encode.

MEASURED RESULT, and the prediction it corrected. The expectation written down in advance
was that the second level would carry the ordering, since the Lévy area
\\iint_{s<t} dX_s \\otimes dX_t is the classical order-sensitive term. It does not. Depth 2
sits at chance (0.49); the separation appears at depth 3 (0.98).

The reason is specific to this path construction and is worth stating, because it is the
kind of thing the universality theorem does not warn you about. With channels
(time, magnitude) the time coordinate is strictly increasing, so the antisymmetric part of
level 2 -- the Lévy area -- reduces to \\int m\\,dt up to boundary terms that vanish when the
path starts and ends at the same magnitude. That integral is the area under the light
curve, which is identical whether the bright peak comes first or second. The area is
order-blind here not by accident but by geometry: a path monotone in one coordinate encloses
no signed area that ordering can change.

Ordering therefore first becomes visible at level 3, in terms of the form
\\iiint dt\\,dm\\,dt, which weight the magnitude excursions by when they happened.

This has a direct consequence for T1 and is the reason the experiment exists: truncating at
depth 2 to save features would discard exactly the information the method is supposed to
provide, and the resulting null result would have been blamed on the astrophysics.

An earlier version of this experiment used a fast-rise/slow-decline pair and its time
reverse. That construction was faulty: reversing time also swaps the first and last
magnitudes, so the classes differed in net change and signature depth 1 scored 0.86 when
theory says it must sit at chance. The discrepancy was a defect in the synthetic data, not
in the method, and the design above removes it.

The three ordering-sensitive features in the baseline (linear trend, rise fraction,
maximum slope) are reported both included and excluded, since including them makes the
baseline able to solve this particular task -- which is the honest result, and it also
shows precisely how narrow that ability is.

Run:  python3 src/order_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features_baseline import band_features, _BAND_FEATURES  # noqa: E402
from features_signature import _log_modulus  # noqa: E402
from signature import signature  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260726
ORDER_BLIND = [f for f in _BAND_FEATURES
               if f not in ("linear_trend", "max_slope", "median_abs_slope",
                            "rise_frac", "fade_frac", "cadence_median", "cadence_iqr")]


PEAK_TIMES = (0.3, 0.7)      # symmetric about the midpoint, so reversal is exact
PEAK_AMPLITUDES = (1.0, 0.5)
PEAK_WIDTH = 0.07


def make_pair(rng: np.random.Generator, n: int, reverse: bool, noise: float
              ) -> tuple[np.ndarray, np.ndarray]:
    """One double-peaked light curve, or the same curve with the peaks swapped.

    Sampling times are irregular and differ between objects, so cadence carries no class
    information. Amplitudes are jittered per object so the classifier cannot key on an
    exact peak height.
    """
    t = np.sort(rng.uniform(0, 1, size=n))
    a1, a2 = PEAK_AMPLITUDES
    jitter = rng.normal(1.0, 0.08, size=2)
    a1, a2 = a1 * jitter[0], a2 * jitter[1]
    if reverse:
        a1, a2 = a2, a1
    t1, t2 = PEAK_TIMES
    shape = (a1 * np.exp(-((t - t1) / PEAK_WIDTH) ** 2)
             + a2 * np.exp(-((t - t2) / PEAK_WIDTH) ** 2))
    m = shape + rng.normal(0, noise, size=n)
    return t, m


def build_dataset(n_objects: int = 800, n_min: int = 20, n_max: int = 60,
                  noise: float = 0.05, seed: int = SEED
                  ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    ts, ms, ys = [], [], []
    for i in range(n_objects):
        n = int(rng.integers(n_min, n_max))
        rev = bool(i % 2)
        t, m = make_pair(rng, n, rev, noise)
        ts.append(t)
        ms.append(m)
        ys.append(int(rev))
    return ts, ms, np.array(ys)


def _cv(X: np.ndarray, y: np.ndarray, folds: int = 5) -> dict[str, float]:
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    lin = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, random_state=SEED))
    trees = HistGradientBoostingClassifier(random_state=SEED)
    return {
        "logistic": round(float(np.mean(cross_val_score(
            lin, X, y, cv=skf, scoring="balanced_accuracy"))), 4),
        "boosted_trees": round(float(np.mean(cross_val_score(
            trees, X, y, cv=skf, scoring="balanced_accuracy"))), 4),
    }


def main() -> int:
    ts, ms, y = build_dataset()
    print(f"{len(y)} synthetic objects, {int(y.sum())} reversed, "
          f"median {int(np.median([len(t) for t in ts]))} points each")

    # Verify the construction actually holds: the magnitude distributions must match.
    pooled_a = np.concatenate([m for m, lab in zip(ms, y) if lab == 0])
    pooled_b = np.concatenate([m for m, lab in zip(ms, y) if lab == 1])
    print(f"pooled magnitude distributions: "
          f"mean {pooled_a.mean():.4f} vs {pooled_b.mean():.4f}, "
          f"std {pooled_a.std():.4f} vs {pooled_b.std():.4f}")

    errs = [np.full_like(m, 0.05) for m in ms]
    full = np.array([[band_features(t, m, e)[k] for k in _BAND_FEATURES]
                     for t, m, e in zip(ts, ms, errs)])
    blind_idx = [_BAND_FEATURES.index(k) for k in ORDER_BLIND]
    blind = full[:, blind_idx]

    results = {
        "summary_order_blind": _cv(blind, y),
        "summary_full": _cv(full, y),
    }
    print(f"\nsummary, order-blind subset ({len(ORDER_BLIND)} features): "
          f"{results['summary_order_blind']}")
    print(f"summary, full set ({len(_BAND_FEATURES)} features): {results['summary_full']}")

    print("\nsignature by truncation depth:")
    for depth in (1, 2, 3, 4):
        feats = np.array([_log_modulus(signature(np.column_stack([t, m]), depth))
                          for t, m in zip(ts, ms)])
        results[f"signature_d{depth}"] = _cv(feats, y)
        results[f"signature_d{depth}"]["n_features"] = int(feats.shape[1])
        print(f"  depth {depth} ({feats.shape[1]:3d} features): "
              f"{results[f'signature_d{depth}']}")

    # Level 1 is the total increment and is order-blind by construction, so it must sit at
    # chance. Some higher level must rise above it, and which one is the finding: with a
    # monotone time channel the level-2 Lévy area degenerates to the area under the curve,
    # so ordering does not appear until level 3.
    blind_acc = results["summary_order_blind"]["boosted_trees"]
    d1 = results["signature_d1"]["boosted_trees"]
    by_depth = {d: results[f"signature_d{d}"]["boosted_trees"] for d in (1, 2, 3, 4)}
    first_above = next((d for d in (1, 2, 3, 4) if by_depth[d] > 0.75), None)
    verdict = {
        "order_blind_at_chance": bool(blind_acc < 0.60),
        "depth1_at_chance": bool(d1 < 0.60),
        "accuracy_by_depth": by_depth,
        "first_depth_above_0.75": first_above,
        "mechanism_confirmed": bool(blind_acc < 0.60 and d1 < 0.60
                                    and first_above is not None),
    }
    print("\norder-blind features: %.3f   signature by depth: %s"
          % (blind_acc, ", ".join(f"d{d}={v:.3f}" for d, v in by_depth.items())))
    print(f"ordering first resolved at depth: {first_above}")
    print(f"mechanism confirmed: {verdict['mechanism_confirmed']}")

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "order_sensitivity.json").write_text(json.dumps({
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "seed": SEED,
        "n_objects": int(len(y)),
        "order_blind_features": ORDER_BLIND,
        "results": results,
        "verdict": verdict,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
