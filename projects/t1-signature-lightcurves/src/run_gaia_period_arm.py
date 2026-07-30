"""The one-feature test: does the catalogued period beat 680 signature coefficients?

This is the sharpest form of the T1 counter-test's argument, and it was missing from the
first run.

The registered prediction was that signatures should lose harder on Gaia periodic variables
than on ZTF transients, because period is the physical discriminant for these classes and
reparameterisation invariance discards exactly the clock that period lives on. The benchmark
tested that indirectly, by comparing gaps. This tests it directly:

    give a classifier ONE number per object -- the catalogued period -- and see whether it
    beats a 680-coefficient signature representation.

If a single feature wins, the argument is not a statistical trend but a structural fact
about what these classes are separated by. If it loses, the account offered for the Gaia
result is wrong and must be replaced.

Four arms, identical folds, identical classifiers:

  period_only        log10(period), one feature
  summary            the hand-crafted variability features used throughout T1
  signature          the best-performing signature configuration from the benchmark
  summary+period     whether period adds to features that already sample the light curve

Restricted to objects with a catalogued period, which is 2,070 of 2,999 -- the classes
without one (long-period and rotational variables in part) are exactly where a period is
hard to define, so the restriction is not neutral and is reported rather than hidden.

Run:  python3 src/run_gaia_period_arm.py [--tag full]
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
from run_benchmark import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260730
BANDS = ("G", "BP", "RP")


def load(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    lc = pd.read_parquet(ROOT / "data" / f"gaia_dr3_variables_{tag}.parquet")
    per = pd.read_csv(ROOT / "data" / f"gaia_dr3_periods_{tag}.csv")
    return lc, per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--mode", default="per_band")
    ap.add_argument("--norm", default="raw_time")
    args = ap.parse_args()

    lc, per = load(args.tag)
    # The light-curve frame keys on source_id; the shared feature code expects `oid`.
    lc = lc.rename(columns={"source_id": "oid"})
    per = per.rename(columns={"source_id": "oid"})
    per = per.dropna(subset=["period_d"])
    per = per[per.period_d > 0]

    all_objects = lc.oid.nunique()
    lc = lc[lc.oid.isin(set(per.oid))]
    kept = lc.oid.nunique()
    print(f"{kept} of {all_objects} objects have a catalogued period "
          f"({kept / max(all_objects, 1):.1%})")
    print(lc.groupby("label").oid.nunique().sort_values(ascending=False).to_string())

    # Which classes lose the most objects to the period requirement? Stated because the
    # restriction is not neutral.
    full_counts = pd.read_parquet(
        ROOT / "data" / f"gaia_dr3_variables_{args.tag}.parquet"
    ).rename(columns={"source_id": "oid"}).groupby("label").oid.nunique()
    kept_counts = lc.groupby("label").oid.nunique()
    retention = (kept_counts / full_counts).dropna().sort_values()
    print("\nper-class retention under the period requirement:")
    print(retention.round(3).to_string())

    print("\nbuilding representations ...")
    summary = feature_frame(lc, bands=BANDS)
    sig = signature_feature_frame(lc, depth=args.depth, mode=args.mode, norm=args.norm,
                                  bands=BANDS, use_lead_lag=True)

    common = summary.index.intersection(sig.index)
    period_map = per.set_index("oid").period_d
    common = common.intersection(period_map.index).sort_values()
    y = summary.loc[common, "label"]

    Xs = summary.loc[common].drop(columns=["label"])
    Xg = sig.loc[common].drop(columns=["label"])
    # One feature. log10 because periods span orders of magnitude across these classes.
    Xp = pd.DataFrame({"log10_period_d": np.log10(period_map.loc[common].to_numpy())},
                      index=common)

    arms = {
        "period_only": Xp,
        "summary": Xs,
        f"signature-{args.mode}-{args.norm}-lead_lag-d{args.depth}": Xg,
        "summary+period": pd.concat([Xs, Xp], axis=1),
    }

    print(f"\nevaluating {len(arms)} arms on {len(common)} objects, "
          f"{y.nunique()} classes, chance = {1 / y.nunique():.4f}")
    results = {}
    for name, X in arms.items():
        res = evaluate(X, y, args.folds)
        results[name] = res
        bt = res["boosted_trees"]
        print(f"  {name:48s} dim={res['n_features']:5d}  "
              f"bal-acc={bt['balanced_accuracy']:.4f}+-{bt['balanced_accuracy_std']:.4f}")

    sig_key = f"signature-{args.mode}-{args.norm}-lead_lag-d{args.depth}"
    p_acc = results["period_only"]["boosted_trees"]["balanced_accuracy"]
    s_acc = results[sig_key]["boosted_trees"]["balanced_accuracy"]
    sm_acc = results["summary"]["boosted_trees"]["balanced_accuracy"]

    verdict = {
        "one_feature_beats_signature": bool(p_acc > s_acc),
        "period_minus_signature": round(p_acc - s_acc, 4),
        "period_features": 1,
        "signature_features": int(results[sig_key]["n_features"]),
        "period_minus_summary": round(p_acc - sm_acc, 4),
        "interpretation": (
            "If one catalogued period outperforms several hundred signature coefficients, "
            "the classes are separated by the clock, which is precisely what "
            "reparameterisation invariance discards. That is a structural statement about "
            "the data, not a statistical trend."),
    }
    print(f"\nperiod ({verdict['period_features']} feature) minus signature "
          f"({verdict['signature_features']} features): {verdict['period_minus_signature']:+.4f}")
    print(f"one feature beats the signature: {verdict['one_feature_beats_signature']}")

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    payload = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "seed": SEED, "tag": args.tag, "folds": args.folds,
        "n_objects": int(len(common)),
        "n_objects_full_sample": int(all_objects),
        "period_retention_overall": round(kept / max(all_objects, 1), 4),
        "period_retention_by_class": {k: round(float(v), 3)
                                      for k, v in retention.items()},
        "class_counts": kept_counts.to_dict(),
        "results": results,
        "verdict": verdict,
    }
    (out / "gaia_period_arm.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame({k: {"balanced_accuracy": v["boosted_trees"]["balanced_accuracy"],
                      "macro_f1": v["boosted_trees"]["macro_f1"],
                      "n_features": v["n_features"]}
                  for k, v in results.items()}).T.to_csv(out / "gaia_period_arm.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
