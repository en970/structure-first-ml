"""Put an interval on T1's headline margins, and separate lead-lag from its dimension.

WHY THIS EXISTS
---------------
`run_gaia_countertest.py` reports three margins that the project README reads as findings:

    P1  gap_baseline_minus_best_signature
    P2  value_of_keeping_duration          raw_time minus unit_time
        value_of_lead_lag                  lead_lag minus no augmentation

Each is computed in `derive_headline` as a difference of two `evaluate()` means. Two things
follow, and they pull in opposite directions, so both have to be said.

The point estimates are sound. `evaluate` builds its folds with
`StratifiedKFold(shuffle=True, random_state=SEED)`, which depends only on the label vector
and the sample count; every variant in the counter-test was measured on the same 2,999 Gaia
(2,375 ZTF) objects in sorted index order. The folds are therefore identical across variants,
and for identical folds mean(A) - mean(B) equals mean(A - B). The published numbers are
genuine paired mean differences.

What is missing is the uncertainty. `evaluate` returns only the mean and standard deviation
per arm and discards the per-fold scores, so no interval on the difference can be recovered
from `outputs/` -- it has to be re-measured. That matters unevenly:

    Gaia  value_of_lead_lag = +0.1130, per-arm fold std 0.0077 and 0.0073
    ZTF   value_of_lead_lag = +0.0246, per-arm fold std 0.0473 and 0.0286

The Gaia margin is an order of magnitude above the fold noise of either arm and will almost
certainly survive. The ZTF margin is smaller than either arm's own spread, so whether it
survives depends entirely on how correlated the two arms are across folds -- which is the one
thing a paired test measures and an unpaired standard deviation cannot tell you. t3's
`leadlag_depth.py` found a spurious +0.0250 at p=0.051 in a setting where the true effect is
provably zero, which is the same size as the ZTF number. That is the coincidence this module
is here to resolve.

THE SECOND DEFECT, WHICH IS LARGER
----------------------------------
The lead-lag margin is confounded with feature count and nothing in the counter-test controls
for it. On Gaia the comparison is:

    signature-per_band-raw_time-d4            90 features   0.7903
    signature-per_band-raw_time-lead_lag-d4 1020 features   0.9033

Lead-lag doubles the channel count, so a depth-4 signature grows eleven-fold. Attributing the
whole +0.113 to "the lead-lag transform exposes quadratic variation" requires that 930 extra
columns of anything would not have helped a gradient-boosted ensemble on their own. That
proposition is testable and has not been tested.

The machinery to test it is already in this repository and is already trusted: the
matched-noise control in `test_complementarity.py`, which was applied to the
summary-plus-signature question and never to this one. This module wires the same control to
this comparison, from both directions:

    plain + noise padding to the lead-lag width   does width alone buy the gain?
    lead-lag randomly projected to the plain width   does the gain survive without the width?

The padding control is the decisive one and R2 is written against it. The projection control
is deliberately ONE-SIDED evidence: a Gaussian projection from 1020 columns to 90 discards
information whether or not the lead-lag transform carries any, so surviving it is strong
evidence for the claim while failing it is weak evidence against. It is reported, and no
verdict rule is hung on it.

VERDICT RULES, FIXED HERE BEFORE THE FIRST FOLD RUNS
----------------------------------------------------
    R1  A margin is reportable only if its 95 percent bootstrap interval on paired folds
        excludes zero.
    R2  The lead-lag margin counts as INFORMATION only if lead_lag beats plain+noise at the
        same width, interval excluding zero. If it does not, the published value_of_lead_lag
        is a statement about feature count and must be restated as one.
    R3  If plain+noise recovers more than half of the published margin over plain, the
        published number is reported as substantially a dimension effect regardless of R2.
    R4  Any margin whose interval includes zero is retracted from the README, and the
        retraction is written into the outputs rather than the number quietly dropped.

Run:  python3 src/audit_headline_margins.py --track gaia
      python3 src/audit_headline_margins.py --track ztf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features_baseline import feature_frame  # noqa: E402
from features_signature import signature_feature_frame  # noqa: E402
from test_complementarity import bootstrap_ci, compare, paired_scores  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260726          # the counter-test seed, deliberately unchanged

TRACKS = {
    "gaia": {"path": "gaia_dr3_variables_full.parquet", "bands": ("G", "BP", "RP"),
             "oid_col": "source_id"},
    "ztf": {"path": "ztf_bts_lightcurves.parquet", "bands": ("g", "r"), "oid_col": None},
}

# The published margins this module is auditing, read from the counter-test output so that
# the comparison is against what was actually reported rather than against a retyped number.
PUBLISHED = ROOT / "outputs" / "gaia_countertest.json"

R3_FRACTION = 0.5        # rule R3: dimension effect threshold, as a fraction of the margin


def load_track(track: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
    cfg = TRACKS[track]
    path = ROOT / "data" / cfg["path"]
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    df = pd.read_parquet(path)
    if cfg["oid_col"]:
        df = df.rename(columns={cfg["oid_col"]: "oid"})
    return df, cfg["bands"]


def published_margins(track: str) -> dict:
    """The numbers being audited, lifted from the counter-test output."""
    if not PUBLISHED.exists():
        return {}
    blob = json.loads(PUBLISHED.read_text())
    head = blob.get(track, {}).get("headline")
    if not head:
        return {}
    out = {}
    if "value_of_lead_lag" in head:
        out["value_of_lead_lag"] = head["value_of_lead_lag"]["value"]
    tc = head.get("time_channel", {})
    if "value_of_keeping_duration" in tc:
        out["value_of_keeping_duration"] = tc["value_of_keeping_duration"]
    if "gap_baseline_minus_best_signature" in head:
        out["gap_baseline_minus_best_signature"] = head["gap_baseline_minus_best_signature"]
    return out


def build_blocks(df: pd.DataFrame, bands: tuple[str, ...]) -> tuple[dict, np.ndarray, dict]:
    """The six feature blocks, all on the same objects in the same order.

    Alignment is done once, on the intersection of every frame's index, so that
    `paired_scores` sees one label vector and one sample count and therefore one set of folds.
    """
    t0 = time.time()
    frames = {
        "summary": feature_frame(df, bands=bands),
        "plain_unit": signature_feature_frame(df, depth=4, mode="per_band", bands=bands,
                                              norm="unit_time", use_lead_lag=False),
        "plain_raw": signature_feature_frame(df, depth=4, mode="per_band", bands=bands,
                                             norm="raw_time", use_lead_lag=False),
        "leadlag_raw": signature_feature_frame(df, depth=4, mode="per_band", bands=bands,
                                               norm="raw_time", use_lead_lag=True),
    }
    common = None
    for f in frames.values():
        common = f.index if common is None else common.intersection(f.index)
    common = common.sort_values()
    y = frames["summary"].loc[common, "label"].to_numpy()

    blocks = {k: np.nan_to_num(f.loc[common].drop(columns=["label"]).to_numpy(float))
              for k, f in frames.items()}

    n_plain, n_ll = blocks["plain_raw"].shape[1], blocks["leadlag_raw"].shape[1]
    rng = np.random.default_rng(SEED)

    # Control one: the plain block widened to the lead-lag width with columns that carry no
    # information. Same construction and scaling as the matched-noise control in
    # test_complementarity.main, so this control is the one the project already trusts.
    pad = n_ll - n_plain
    noise = rng.normal(0.0, blocks["leadlag_raw"].std(axis=0, keepdims=True)[:, :pad] + 1e-9,
                       size=(len(y), pad))
    blocks["plain_raw+noise"] = np.hstack([blocks["plain_raw"], noise])

    # Control two: the lead-lag block narrowed to the plain width by Gaussian random
    # projection. This is the same question from the other side -- if the information is
    # there, a random projection to 90 columns keeps most of it; if the gain was width, it
    # does not survive.
    proj = rng.normal(0.0, 1.0 / np.sqrt(n_plain), size=(n_ll, n_plain))
    blocks["leadlag_proj"] = blocks["leadlag_raw"] @ proj

    meta = {"n_objects": int(len(y)), "n_classes": int(len(np.unique(y))),
            "widths": {k: int(v.shape[1]) for k, v in blocks.items()},
            "build_seconds": round(time.time() - t0, 1)}
    return blocks, y, meta


# The contrasts, named so the output can be read without the code. Each is (a, b): b minus a.
CONTRASTS = {
    "value_of_lead_lag__published_form": ("plain_raw", "leadlag_raw"),
    "dimension_alone": ("plain_raw", "plain_raw+noise"),
    "lead_lag_over_matched_width": ("plain_raw+noise", "leadlag_raw"),
    "lead_lag_at_plain_width": ("plain_raw", "leadlag_proj"),
    "value_of_keeping_duration": ("plain_unit", "plain_raw"),
    "gap_baseline_minus_plain": ("plain_raw", "summary"),
    "gap_baseline_minus_leadlag": ("leadlag_raw", "summary"),
}


def verdicts(results: dict, pub: dict) -> dict:
    """Apply R1-R4 exactly as written in the module docstring."""
    def excludes_zero(name):
        lo, hi = results[name]["ci95"]
        return lo > 0 or hi < 0

    out = {"R1_reportable": {k: bool(excludes_zero(k)) for k in results}}

    ll_pub = results["value_of_lead_lag__published_form"]["mean_difference"]
    matched = results["lead_lag_over_matched_width"]
    dim = results["dimension_alone"]["mean_difference"]

    out["R2_lead_lag_is_information"] = {
        "lead_lag_over_matched_width": matched["mean_difference"],
        "ci95": matched["ci95"],
        "passes": bool(matched["ci95"][0] > 0),
        "meaning": ("lead-lag beats an equally wide block of noise, so the margin is "
                    "information" if matched["ci95"][0] > 0 else
                    "lead-lag does NOT beat an equally wide block of noise; the published "
                    "value_of_lead_lag must be restated as a statement about feature count"),
    }
    # A negative dimension_alone means the padding HURTS, which is the clean outcome: extra
    # width is not doing the work. Reporting it as a "fraction recovered" would be misleading,
    # so the sign is named instead.
    frac = round(dim / ll_pub, 3) if ll_pub else None
    out["R3_substantially_dimension"] = {
        "dimension_alone": dim,
        "published_margin": ll_pub,
        "fraction_recovered_by_width": frac,
        "width_alone_helps": bool(dim > 0),
        "threshold": R3_FRACTION,
        "triggered": bool(ll_pub and dim > 0 and dim / ll_pub > R3_FRACTION),
        "reading": ("padding the plain block to the lead-lag width HURTS it, so the margin "
                    "is not bought by feature count" if dim <= 0 else
                    f"padding alone recovers {frac} of the published margin"),
    }
    out["R4_retractions"] = [k for k in results if not excludes_zero(k)]

    # Did the re-measured point estimate reproduce the published one? A drift here would mean
    # something changed in the pipeline since the counter-test ran, and every other number in
    # this file would need re-reading before it could be trusted.
    out["reproduction_check"] = {}
    for name, key in (("value_of_lead_lag__published_form", "value_of_lead_lag"),
                      ("value_of_keeping_duration", "value_of_keeping_duration")):
        if key in pub:
            got = results[name]["mean_difference"]
            out["reproduction_check"][key] = {
                "published": pub[key], "remeasured": got,
                "absolute_difference": round(abs(got - pub[key]), 4),
                # 50 paired folds against the published 5, so exact agreement is not expected;
                # a gap beyond 0.02 means more than resampling noise and is flagged.
                "consistent": bool(abs(got - pub[key]) < 0.02),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", required=True, choices=list(TRACKS))
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    df, bands = load_track(args.track)
    print(f"[{args.track}] building feature blocks ...", flush=True)
    blocks, y, meta = build_blocks(df, bands)
    print(f"  {meta['n_objects']} objects, {meta['n_classes']} classes, "
          f"widths {meta['widths']}  ({meta['build_seconds']}s)")

    n_folds = args.splits * args.repeats
    print(f"[{args.track}] {n_folds} paired folds over {len(blocks)} blocks "
          f"(this is the slow part) ...", flush=True)
    t0 = time.time()
    scores = paired_scores(blocks, y, args.splits, args.repeats)
    elapsed = round(time.time() - t0, 1)
    for name, s in scores.items():
        print(f"    {name:20s} {s.mean():.4f} +- {s.std():.4f}")

    results = {name: compare(scores, a, b) for name, (a, b) in CONTRASTS.items()}
    pub = published_margins(args.track)
    verdict = verdicts(results, pub)

    print(f"\n[{args.track}] contrasts (95% bootstrap CI on {n_folds} paired folds):")
    for name, r in results.items():
        mark = " " if (r["ci95"][0] > 0 or r["ci95"][1] < 0) else "*"
        print(f"  {mark} {name:38s} {r['mean_difference']:+.4f}  "
              f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  p={r['wilcoxon_p']}  "
              f"{r['folds_improved']}/{r['n_folds']}")
    print("  (* = interval includes zero, retracted under R4)")

    print(f"\n  R2 lead-lag is information: {verdict['R2_lead_lag_is_information']['passes']}")
    print(f"     {verdict['R2_lead_lag_is_information']['meaning']}")
    print(f"  R3 substantially a dimension effect: "
          f"{verdict['R3_substantially_dimension']['triggered']}")
    print(f"     {verdict['R3_substantially_dimension']['reading']}")
    if verdict["R4_retractions"]:
        print(f"  R4 retract: {verdict['R4_retractions']}")

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    path = out / f"audit_headline_margins_{args.track}.json"
    path.write_text(json.dumps({
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "track": args.track, "seed": SEED,
        "splits": args.splits, "repeats": args.repeats, "n_folds": n_folds,
        "sample": meta,
        "published_margins_being_audited": pub,
        "per_block_mean": {k: round(float(v.mean()), 4) for k, v in scores.items()},
        "per_block_std": {k: round(float(v.std()), 4) for k, v in scores.items()},
        "contrasts": results,
        "verdicts": verdict,
        "elapsed_s": elapsed,
        "_fold_scores": {k: [round(float(x), 6) for x in v] for k, v in scores.items()},
    }, indent=2))
    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
