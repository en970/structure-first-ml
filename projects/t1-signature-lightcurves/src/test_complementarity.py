"""Is the gain from adding signature features to the baseline real, or fold noise?

The headline benchmark reports that concatenating signature features with the hand-crafted
baseline scores above the baseline alone. The difference is of the same order as the
spread across folds, so on its own that comparison establishes nothing. This module runs
the test properly.

Design:

  Repeated stratified cross-validation. Five folds repeated ten times with different
  shuffles, giving fifty paired measurements instead of five.

  Paired comparison on identical splits. Both representations see exactly the same train
  and test indices in every fold, so the fold-to-fold variance -- which is large and is
  driven by which rare objects land in which fold -- cancels in the difference. Comparing
  two independent runs would leave that variance in place and is the usual way this
  question gets answered badly.

  A Wilcoxon signed-rank test on the paired differences, which assumes neither normality
  nor equal variances, alongside the mean difference and a bootstrap interval. The test is
  one of several possible; the effect size and interval are what should be read.

  A permutation control. The same procedure is run with the signature block replaced by
  Gaussian noise of identical shape. If adding 680 columns of noise also produces an
  apparent gain, then the gain is a property of the classifier's capacity rather than of
  the features, and the whole comparison is void.

That last control is the one that matters. Concatenating any large block of columns to a
small feature set changes the model's behaviour, and a gradient-boosted tree ensemble given
more candidate splits can improve for reasons having nothing to do with what the columns
contain.

Run:  python3 src/test_complementarity.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features_baseline import feature_frame  # noqa: E402
from features_signature import signature_feature_frame  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260726


def paired_scores(blocks: dict[str, np.ndarray], y: np.ndarray, n_splits: int,
                  n_repeats: int) -> dict[str, np.ndarray]:
    """Balanced accuracy per fold for each feature block, on identical splits."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=SEED)
    scores = {k: [] for k in blocks}
    for tr, te in rskf.split(next(iter(blocks.values())), y):
        for name, X in blocks.items():
            model = HistGradientBoostingClassifier(random_state=SEED,
                                                   class_weight="balanced")
            model.fit(X[tr], y[tr])
            scores[name].append(balanced_accuracy_score(y[te], model.predict(X[te])))
    return {k: np.asarray(v) for k, v in scores.items()}


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, alpha: float = 0.05
                 ) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    means = rng.choice(diff, size=(n_boot, diff.size), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def compare(scores: dict[str, np.ndarray], a: str, b: str) -> dict:
    """Paired comparison of block b against block a."""
    diff = scores[b] - scores[a]
    lo, hi = bootstrap_ci(diff)
    try:
        stat, p = wilcoxon(diff)
        p = float(p)
    except ValueError:  # all differences identical
        p = 1.0
    return {
        "mean_" + a: round(float(scores[a].mean()), 4),
        "mean_" + b: round(float(scores[b].mean()), 4),
        "mean_difference": round(float(diff.mean()), 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "wilcoxon_p": float(f"{p:.3g}"),
        "folds_improved": int((diff > 0).sum()),
        "n_folds": int(diff.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--mode", default="per_band")
    ap.add_argument("--norm", default="raw_time")
    args = ap.parse_args()

    path = ROOT / "data" / "ztf_bts_lightcurves.parquet"
    if not path.exists():
        print(f"missing {path}; run src/fetch_ztf_bts.py first", file=sys.stderr)
        return 1
    df = pd.read_parquet(path)

    summary = feature_frame(df)
    sig = signature_feature_frame(df, depth=args.depth, mode=args.mode, norm=args.norm,
                                  use_lead_lag=True)
    common = summary.index.intersection(sig.index).sort_values()
    y = summary.loc[common, "label"].to_numpy()
    Xs = np.nan_to_num(summary.loc[common].drop(columns=["label"]).to_numpy(float))
    Xg = np.nan_to_num(sig.loc[common].drop(columns=["label"]).to_numpy(float))
    print(f"{len(common)} objects, summary {Xs.shape[1]} features, "
          f"signature {Xg.shape[1]} features")

    # Noise block of identical shape, scaled to the signature block's column variance, so
    # that the control differs from the real block only in its information content.
    rng = np.random.default_rng(SEED)
    Xn = rng.normal(0.0, Xg.std(axis=0, keepdims=True) + 1e-9, size=Xg.shape)

    blocks = {
        "summary": Xs,
        "summary+signature": np.hstack([Xs, Xg]),
        "summary+noise": np.hstack([Xs, Xn]),
        "signature": Xg,
    }
    n_folds = args.splits * args.repeats
    print(f"running {n_folds} paired folds over {len(blocks)} feature blocks ...")
    scores = paired_scores(blocks, y, args.splits, args.repeats)

    for name, s in scores.items():
        print(f"  {name:20s} {s.mean():.4f} +- {s.std():.4f}")

    real = compare(scores, "summary", "summary+signature")
    control = compare(scores, "summary", "summary+noise")
    alone = compare(scores, "summary", "signature")

    print("\nsummary + signature vs summary:")
    print(f"  difference {real['mean_difference']:+.4f}  "
          f"95% CI [{real['ci95'][0]:+.4f}, {real['ci95'][1]:+.4f}]  "
          f"p={real['wilcoxon_p']}  improved in {real['folds_improved']}/{real['n_folds']} folds")
    print("noise control, summary + noise vs summary:")
    print(f"  difference {control['mean_difference']:+.4f}  "
          f"95% CI [{control['ci95'][0]:+.4f}, {control['ci95'][1]:+.4f}]  "
          f"p={control['wilcoxon_p']}  improved in {control['folds_improved']}/{control['n_folds']} folds")

    # The signature block only counts as complementary if it beats the baseline AND the
    # noise block of the same width does not.
    verdict = {
        "signature_adds_over_baseline": bool(real["ci95"][0] > 0),
        "noise_control_clean": bool(control["ci95"][0] <= 0),
        "gain_exceeds_noise_control": bool(
            real["mean_difference"] > control["mean_difference"]),
        "complementary": bool(real["ci95"][0] > 0 and control["ci95"][0] <= 0),
    }
    print(f"\ncomplementary: {verdict['complementary']}")

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "complementarity.json").write_text(json.dumps({
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "seed": SEED, "n_objects": int(len(common)),
        "splits": args.splits, "repeats": args.repeats,
        "signature_config": {"mode": args.mode, "norm": args.norm,
                             "depth": args.depth, "lead_lag": True},
        "per_block_mean": {k: round(float(v.mean()), 4) for k, v in scores.items()},
        "per_block_std": {k: round(float(v.std()), 4) for k, v in scores.items()},
        "summary_vs_summary_plus_signature": real,
        "summary_vs_summary_plus_noise": control,
        "summary_vs_signature_alone": alone,
        "verdict": verdict,
    }, indent=2))
    pd.DataFrame(scores).to_csv(out / "complementarity_folds.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
