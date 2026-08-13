"""Is the lead-lag margin quadratic variation, and is t3's depth-2 identity true on photometry?

WHAT t3 PROVED, AND WHAT IT DOES NOT COVER
------------------------------------------
`t3-signature-theory` established in closed form (verified numerically to 8.9e-16) that for a
lead-lag transformed path, everything the transform adds AT DEPTH <= 2 is carried by the
realised (co)variation scalars of the underlying path. For a per-band path with channels
(t, m) those are three numbers per band:

    Qtt = sum (dt)^2      Qmm = sum (dm)^2      Qtm = sum (dt)(dm)

Three bands on Gaia gives nine scalars; two bands on ZTF gives six. They are order-blind: any
permutation of the increments leaves them unchanged.

That result is exact and it is bounded. It says nothing about depth 3 or depth 4, and the
counter-test's headline margin is measured at DEPTH 4, where lead-lag also contributes terms
the theorem does not reach. So this module asks two different questions and keeps them apart,
because collapsing them would be the same error the counter-test made with feature width.

    Q1 (the theorem, on real data)  At DEPTH 2, where the identity is exact, does
       plain + Q match lead-lag? This is an EQUALITY prediction, which is far sharper than a
       directional one: the theorem says these two blocks carry the same information, so
       neither should beat the other outside noise.

    Q2 (the headline, at depth 4)  How much of the depth-4 lead-lag margin do nine scalars
       recover? A large fraction means the counter-test's best signature arm is substantially
       measuring quadratic variation rather than path ordering, and the README's account of
       its own result is wrong. A small fraction means the depth-3 and depth-4 lead-lag terms
       are doing the work, which the theorem does not explain and which then becomes the
       interesting open question.

Both questions are asked against a matched-width noise control, for the reason
`audit_headline_margins.py` sets out: adding columns to a gradient-boosted ensemble changes
its behaviour whatever the columns contain, and nine columns is small but not nothing.

PREDICTIONS, FIXED HERE BEFORE THE FIRST FOLD RUNS
--------------------------------------------------
    P-A  At depth 2, |plain+Q minus leadlag| has a 95% interval containing zero.
         t3's identity is exact, so a resolved difference either way falsifies its transfer
         to real photometry.
    P-B  At depth 2, plain+Q beats plain, interval excluding zero, and beats plain+noise(9).
         If nine informative scalars cannot beat nine noise columns, the depth-2 arm is too
         insensitive to test anything and Q1 is reported as unresolved rather than passed.
    P-C  At depth 4, the fraction of the lead-lag margin recovered by Q is reported with an
         interval. No threshold is pre-set for "large", because the interesting quantity is
         the number itself; but the sign and the interval are pre-committed as the headline.

Run:  python3 src/quadratic_variation_ablation.py --track gaia
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
from features_signature import _log_modulus, build_path  # noqa: E402
from test_complementarity import compare, paired_scores  # noqa: E402
from audit_headline_margins import TRACKS, load_track  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260726


def quadratic_variation_frame(df: pd.DataFrame, bands: tuple[str, ...],
                              norm: str = "raw_time") -> pd.DataFrame:
    """Qtt, Qmm, Qtm per band, on exactly the path the signature is computed from.

    The path is built with the same `build_path` call the signature features use, so this
    block cannot differ from the signature arm by any preparation choice -- only by what is
    computed from the path. Scaled with the same `_log_modulus` the signature features use,
    so the two blocks reach the classifier on comparable scales.
    """
    rows, index, labels = [], [], []
    for oid, lc in df.groupby("oid", sort=True):
        paths = build_path(lc, mode="per_band", bands=bands, with_time=True, norm=norm)
        vals = []
        for p in paths:                      # one path per band, columns (t, m)
            d = np.diff(p, axis=0)
            if d.size == 0:
                vals += [0.0, 0.0, 0.0]
                continue
            dt, dm = d[:, 0], d[:, 1]
            vals += [float(dt @ dt), float(dm @ dm), float(dt @ dm)]
        rows.append(vals)
        index.append(oid)
        labels.append(lc.label.iloc[0])
    cols = [f"Q{s}_{b}" for b in bands for s in ("tt", "mm", "tm")]
    out = pd.DataFrame(_log_modulus(np.asarray(rows)), index=pd.Index(index, name="oid"),
                       columns=cols)
    out["label"] = labels
    return out


def build_blocks(df: pd.DataFrame, bands: tuple[str, ...]) -> tuple[dict, np.ndarray, dict]:
    from features_signature import signature_feature_frame

    t0 = time.time()
    sig = lambda depth, ll: signature_feature_frame(  # noqa: E731
        df, depth=depth, mode="per_band", bands=bands, norm="raw_time", use_lead_lag=ll)
    frames = {
        "plain_d2": sig(2, False),
        "leadlag_d2": sig(2, True),
        "plain_d4": sig(4, False),
        "leadlag_d4": sig(4, True),
        "Q": quadratic_variation_frame(df, bands),
    }
    common = None
    for f in frames.values():
        common = f.index if common is None else common.intersection(f.index)
    common = common.sort_values()
    y = frames["plain_d2"].loc[common, "label"].to_numpy()
    arr = {k: np.nan_to_num(f.loc[common].drop(columns=["label"]).to_numpy(float))
           for k, f in frames.items()}

    n_q = arr["Q"].shape[1]
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0.0, arr["Q"].std(axis=0, keepdims=True) + 1e-9, size=(len(y), n_q))

    blocks = {
        "plain_d2": arr["plain_d2"],
        "plain_d2+Q": np.hstack([arr["plain_d2"], arr["Q"]]),
        "plain_d2+noise": np.hstack([arr["plain_d2"], noise]),
        "leadlag_d2": arr["leadlag_d2"],
        "plain_d4": arr["plain_d4"],
        "plain_d4+Q": np.hstack([arr["plain_d4"], arr["Q"]]),
        "plain_d4+noise": np.hstack([arr["plain_d4"], noise]),
        "leadlag_d4": arr["leadlag_d4"],
    }
    meta = {"n_objects": int(len(y)), "n_classes": int(len(np.unique(y))),
            "n_quadratic_variation_scalars": int(n_q),
            "widths": {k: int(v.shape[1]) for k, v in blocks.items()},
            "build_seconds": round(time.time() - t0, 1)}
    return blocks, y, meta


CONTRASTS = {
    # Q1: the theorem, where it is exact
    "d2__Q_over_plain": ("plain_d2", "plain_d2+Q"),
    "d2__noise_over_plain": ("plain_d2", "plain_d2+noise"),
    "d2__leadlag_over_plain": ("plain_d2", "leadlag_d2"),
    "d2__identity_gap_leadlag_minus_plainQ": ("plain_d2+Q", "leadlag_d2"),
    # Q2: the headline, at the depth it was measured
    "d4__Q_over_plain": ("plain_d4", "plain_d4+Q"),
    "d4__noise_over_plain": ("plain_d4", "plain_d4+noise"),
    "d4__leadlag_over_plain": ("plain_d4", "leadlag_d4"),
    "d4__residual_leadlag_minus_plainQ": ("plain_d4+Q", "leadlag_d4"),
}


def verdicts(r: dict) -> dict:
    spans_zero = lambda k: r[k]["ci95"][0] <= 0 <= r[k]["ci95"][1]  # noqa: E731

    q_d2 = r["d2__Q_over_plain"]["mean_difference"]
    noise_d2 = r["d2__noise_over_plain"]["mean_difference"]
    ll_d4 = r["d4__leadlag_over_plain"]["mean_difference"]
    q_d4 = r["d4__Q_over_plain"]["mean_difference"]

    out = {
        "P_A_depth2_identity_holds": {
            "gap": r["d2__identity_gap_leadlag_minus_plainQ"]["mean_difference"],
            "ci95": r["d2__identity_gap_leadlag_minus_plainQ"]["ci95"],
            "holds": bool(spans_zero("d2__identity_gap_leadlag_minus_plainQ")),
            "meaning": ("plain+Q and lead-lag are indistinguishable at depth 2, as t3's "
                        "identity requires" if spans_zero("d2__identity_gap_leadlag_minus_plainQ")
                        else "plain+Q and lead-lag differ at depth 2; t3's identity does not "
                             "transfer to real photometry as stated, and that is the finding"),
        },
        "P_B_depth2_arm_is_sensitive": {
            "Q_over_plain": q_d2, "noise_over_plain": noise_d2,
            "passes": bool(r["d2__Q_over_plain"]["ci95"][0] > 0 and q_d2 > noise_d2),
            "meaning": ("the depth-2 arm can resolve nine informative columns from nine noise "
                        "columns, so P-A is a real test"
                        if r["d2__Q_over_plain"]["ci95"][0] > 0 and q_d2 > noise_d2 else
                        "the depth-2 arm cannot resolve nine informative columns from nine "
                        "noise columns; P-A is UNRESOLVED, not passed"),
        },
        "P_C_depth4_fraction_recovered": {
            "leadlag_margin": ll_d4,
            "Q_margin": q_d4,
            "fraction": round(q_d4 / ll_d4, 3) if ll_d4 else None,
            "residual_after_Q": r["d4__residual_leadlag_minus_plainQ"]["mean_difference"],
            "residual_ci95": r["d4__residual_leadlag_minus_plainQ"]["ci95"],
            "residual_resolved": bool(not spans_zero("d4__residual_leadlag_minus_plainQ")),
        },
        "intervals_spanning_zero": [k for k in r if spans_zero(k)],
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", required=True, choices=list(TRACKS))
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    df, bands = load_track(args.track)
    print(f"[{args.track}] building blocks ...", flush=True)
    blocks, y, meta = build_blocks(df, bands)
    print(f"  {meta['n_objects']} objects, {meta['n_quadratic_variation_scalars']} "
          f"(co)variation scalars, widths {meta['widths']}  ({meta['build_seconds']}s)")

    n_folds = args.splits * args.repeats
    print(f"[{args.track}] {n_folds} paired folds over {len(blocks)} blocks ...", flush=True)
    t0 = time.time()
    scores = paired_scores(blocks, y, args.splits, args.repeats)
    elapsed = round(time.time() - t0, 1)
    for name, s in scores.items():
        print(f"    {name:18s} {s.mean():.4f} +- {s.std():.4f}")

    results = {name: compare(scores, a, b) for name, (a, b) in CONTRASTS.items()}
    verdict = verdicts(results)

    print(f"\n[{args.track}] contrasts (95% bootstrap CI on {n_folds} paired folds):")
    for name, r in results.items():
        mark = " " if (r["ci95"][0] > 0 or r["ci95"][1] < 0) else "*"
        print(f"  {mark} {name:42s} {r['mean_difference']:+.4f}  "
              f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  p={r['wilcoxon_p']}")
    print("  (* = interval includes zero)")

    print(f"\n  P-A depth-2 identity holds: {verdict['P_A_depth2_identity_holds']['holds']}")
    print(f"      {verdict['P_A_depth2_identity_holds']['meaning']}")
    print(f"  P-B depth-2 arm is sensitive: "
          f"{verdict['P_B_depth2_arm_is_sensitive']['passes']}")
    print(f"      {verdict['P_B_depth2_arm_is_sensitive']['meaning']}")
    pc = verdict["P_C_depth4_fraction_recovered"]
    print(f"  P-C at depth 4: lead-lag margin {pc['leadlag_margin']:+.4f}, "
          f"Q recovers {pc['Q_margin']:+.4f} (fraction {pc['fraction']})")
    print(f"      residual after Q {pc['residual_after_Q']:+.4f} "
          f"{pc['residual_ci95']}  resolved={pc['residual_resolved']}")

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    path = out / f"quadratic_variation_ablation_{args.track}.json"
    path.write_text(json.dumps({
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "track": args.track, "seed": SEED,
        "splits": args.splits, "repeats": args.repeats, "n_folds": n_folds,
        "sample": meta,
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
