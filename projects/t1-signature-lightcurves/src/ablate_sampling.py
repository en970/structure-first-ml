"""Degrade the sampling and measure which representation survives it.

This is the actual experiment. The headline accuracy on full light curves says little,
because every representation does reasonably well when the data are dense. The claim under
test is about *where* signatures should help:

    the signature representation should gain most where sampling is sparsest and most
    irregular, and least where light curves are dense.

So the light curves are thinned systematically and each representation is re-evaluated at
each level. Two thinning regimes are used at matched retention fractions, which is the
point of the design: they remove the same number of observations but destroy different
structure.

  random       Each observation is dropped independently. Gaps are short and scattered.
  blocked      Contiguous runs of observations are dropped, imitating weather outages and
               seasonal visibility windows. Gaps are long and structured.

If the two regimes produce the same degradation curve, then gap *structure* does not
matter and only the number of observations does -- which would undercut a central part of
the motivation. If blocked thinning hurts the interpolation-dependent baselines more than
it hurts signatures, the motivation survives.

Run:  python3 src/ablate_sampling.py [--fractions 1.0 0.7 0.5 0.3 0.2] [--folds 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features_baseline import feature_frame  # noqa: E402
from features_signature import signature_feature_frame  # noqa: E402
from run_benchmark import _align, evaluate, minirocket_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260726
MIN_KEEP = 6  # never thin an object below this many points per band


def thin_random(lc: pd.DataFrame, frac: float, rng: np.random.Generator) -> pd.DataFrame:
    """Drop observations independently, per band."""
    keep = []
    for _, sub in lc.groupby("band"):
        n = len(sub)
        k = max(MIN_KEEP, int(round(n * frac)))
        if k >= n:
            keep.append(sub)
            continue
        idx = rng.choice(n, size=k, replace=False)
        keep.append(sub.iloc[np.sort(idx)])
    return pd.concat(keep)


def thin_blocked(lc: pd.DataFrame, frac: float, rng: np.random.Generator) -> pd.DataFrame:
    """Remove contiguous blocks of observations, imitating weather and season gaps."""
    keep = []
    for _, sub in lc.groupby("band"):
        sub = sub.sort_values("mjd")
        n = len(sub)
        k = max(MIN_KEEP, int(round(n * frac)))
        if k >= n:
            keep.append(sub)
            continue
        n_drop = n - k
        mask = np.ones(n, dtype=bool)
        # Remove in a few contiguous chunks rather than one, so that the result is a
        # gappy season rather than a truncated one.
        n_blocks = max(1, min(3, n_drop // 3))
        sizes = np.full(n_blocks, n_drop // n_blocks)
        sizes[: n_drop % n_blocks] += 1
        for size in sizes:
            avail = np.flatnonzero(mask)
            if len(avail) <= MIN_KEEP or size <= 0:
                break
            size = int(min(size, len(avail) - MIN_KEEP))
            start = int(rng.integers(0, max(1, len(avail) - size)))
            mask[avail[start:start + size]] = False
        keep.append(sub[mask])
    return pd.concat(keep)


def thin_dataset(df: pd.DataFrame, frac: float, regime: str, seed: int) -> pd.DataFrame:
    if frac >= 1.0:
        return df
    rng = np.random.default_rng(seed)
    fn = thin_random if regime == "random" else thin_blocked
    return (df.groupby("oid", group_keys=False)
              .apply(lambda lc: fn(lc, frac, rng), include_groups=True)
              .reset_index(drop=True))


def representations(df: pd.DataFrame, depth: int, mode: str, norm: str
                    ) -> dict[str, pd.DataFrame]:
    reps = {
        "summary": feature_frame(df),
        f"signature-{mode}-d{depth}": signature_feature_frame(
            df, depth=depth, mode=mode, norm=norm),
    }
    mr = minirocket_features(df)
    if mr is not None:
        reps["minirocket"] = mr
    return reps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", type=float, nargs="+",
                    default=[1.0, 0.7, 0.5, 0.35, 0.25, 0.15])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--mode", default="per_band")
    ap.add_argument("--norm", default="raw_time",
                    help="channel preparation; see features_signature.NORMALISATIONS")
    args = ap.parse_args()

    path = ROOT / "data" / "ztf_bts_lightcurves.parquet"
    if not path.exists():
        print(f"missing {path}; run src/fetch_ztf_bts.py first", file=sys.stderr)
        return 1
    df = pd.read_parquet(path)
    print(f"{df.oid.nunique()} objects, {len(df)} detections")

    records = []
    for regime in ("random", "blocked"):
        for frac in args.fractions:
            if frac >= 1.0 and regime == "blocked":
                continue  # the full sample is identical under both regimes
            thinned = thin_dataset(df, frac, regime, SEED)
            n_med = int(thinned.groupby("oid").size().median())
            print(f"\n{regime} frac={frac:.2f}: {len(thinned)} detections "
                  f"(median {n_med} per object)")
            reps = representations(thinned, args.depth, args.mode, args.norm)
            for name, frame in reps.items():
                (X,), y = _align(frame)
                res = evaluate(X, y, args.folds)
                bal = res["boosted_trees"]["balanced_accuracy"]
                std = res["boosted_trees"]["balanced_accuracy_std"]
                print(f"  {name:28s} bal-acc={bal:.4f}+-{std:.4f}")
                records.append({
                    "regime": regime, "fraction": frac, "representation": name,
                    "median_detections": n_med,
                    "balanced_accuracy": bal, "balanced_accuracy_std": std,
                    "macro_f1": res["boosted_trees"]["macro_f1"],
                    "logistic_balanced_accuracy": res["logistic"]["balanced_accuracy"],
                    "n_features": res["n_features"],
                })

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    tab = pd.DataFrame(records)
    # the full-sample row applies to both regimes
    full = tab[tab.fraction >= 1.0].copy()
    if not full.empty:
        full["regime"] = "blocked"
        tab = pd.concat([tab, full], ignore_index=True)
    tab.to_csv(out / "ablation_sampling.csv", index=False)
    (out / "ablation_sampling.json").write_text(json.dumps({
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "seed": SEED, "folds": args.folds, "depth": args.depth, "mode": args.mode, "norm": args.norm,
        "min_keep_per_band": MIN_KEEP,
        "records": records,
    }, indent=2))

    print("\n" + tab.pivot_table(index=["representation", "regime"], columns="fraction",
                                 values="balanced_accuracy").round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
