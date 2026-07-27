"""Consensus clustering across orbital dissimilarity criteria.

The method the field's own review asks for and nobody has run on this archive: cluster the
orbits independently under several classical D-criteria, then keep only the structure that
survives all of them. Stability across metrics is the detection statistic; a group that
appears under one criterion and not the others is an artefact of that criterion's
particular weighting, not a stream.

Why this is not just "run DBSCAN four times". Each criterion weights eccentricity,
perihelion distance, inclination and perihelion orientation differently, and the classical
association thresholds (~0.05-0.2) are not comparable across them. Rather than hand-tuning
four thresholds, each criterion is calibrated to a common false-positive rate against a
sporadic null, so that "close under D_SH" and "close under D_D" mean the same thing
statistically before any consensus is taken.

Scale. Pairwise distance over 2.1 million meteors is 4.6e12 pairs and is not computable.
The archive is therefore processed in overlapping solar-longitude windows, which is
physically motivated rather than a convenience: a stream is active over a bounded range of
solar longitude, so meteors separated by months cannot belong to the same shower encounter
regardless of orbital similarity. Windows overlap so that a stream straddling a boundary is
not cut in half.

The null. The sporadic background is structured -- helion, antihelion, apex and toroidal
sources -- so a uniform null would declare almost everything significant. The null here
resamples within the observed distribution while destroying the fine-scale correlations
that a stream produces, following the spirit of the KDE-based nulls used in the recent
literature (Shober & Vaubaillon 2024; Shober 2026).

Run:  python3 src/consensus_cluster.py [--window 2.0] [--step 1.0] [--min-cluster 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dcriteria import d_d, d_h, d_sh  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260727
CRITERIA = ("d_sh", "d_d", "d_h")
FN = {"d_sh": d_sh, "d_d": d_d, "d_h": d_h}


def orbit_cols(df: pd.DataFrame) -> tuple:
    return (df["q"].to_numpy(), df["e"].to_numpy(), df["i"].to_numpy(),
            df["node"].to_numpy(), df["peri"].to_numpy())


def pairwise(df: pd.DataFrame, criterion: str) -> np.ndarray:
    q, e, i, node, peri = orbit_cols(df)
    return FN[criterion](q, e, i, node, peri, q, e, i, node, peri)


def sporadic_null_threshold(df: pd.DataFrame, criterion: str, fpr: float,
                            rng: np.random.Generator, n_draw: int = 4000) -> float:
    """Calibrate a distance threshold to a common false-positive rate.

    The null pairs are built by shuffling each orbital element independently across the
    window. That preserves every one-dimensional marginal distribution -- so the helion
    and antihelion concentrations, the eccentricity distribution and the inclination
    distribution all survive -- while destroying the joint structure that makes a stream a
    stream. The threshold is the `fpr` quantile of null pair distances, so each criterion
    admits the same fraction of chance pairs before any consensus is taken.
    """
    n = len(df)
    if n < 20:
        return np.nan
    shuffled = pd.DataFrame({
        col: rng.permutation(df[col].to_numpy()) for col in ("q", "e", "i", "node", "peri")
    })
    idx_a = rng.integers(0, n, size=min(n_draw, n * 4))
    idx_b = rng.integers(0, n, size=idx_a.size)
    keep = idx_a != idx_b
    idx_a, idx_b = idx_a[keep], idx_b[keep]

    a, b = shuffled.iloc[idx_a], shuffled.iloc[idx_b]
    qa, ea, ia, na_, pa = orbit_cols(a)
    qb, eb, ib, nb, pb = orbit_cols(b)
    # diagonal of the paired block: distance of pair k, not the full matrix
    d = np.array([FN[criterion](qa[k:k + 1], ea[k:k + 1], ia[k:k + 1], na_[k:k + 1],
                                pa[k:k + 1], qb[k:k + 1], eb[k:k + 1], ib[k:k + 1],
                                nb[k:k + 1], pb[k:k + 1]).ravel()[0]
                  for k in range(len(qa))])
    return float(np.quantile(d, fpr))


def cluster_window(df: pd.DataFrame, thresholds: dict, min_cluster: int) -> pd.DataFrame:
    """Cluster under each criterion; return per-meteor labels, one column each."""
    out = pd.DataFrame(index=df.index)
    for crit in CRITERIA:
        eps = thresholds[crit]
        if not np.isfinite(eps):
            out[crit] = -1
            continue
        dist = pairwise(df, crit)
        labels = DBSCAN(eps=eps, min_samples=min_cluster,
                        metric="precomputed").fit_predict(dist)
        out[crit] = labels
    return out


def consensus_groups(labels: pd.DataFrame, min_cluster: int) -> np.ndarray:
    """Group meteors that share a cluster under EVERY criterion.

    Two meteors are consensus-linked when all criteria place them in the same (non-noise)
    cluster. The consensus label is then the identity of that intersection: meteors are
    grouped by their full label tuple, and any meteor called noise by any single criterion
    is excluded. This is the strict reading of cross-metric stability, deliberately so --
    the point of the track is that a candidate surviving one metric is not evidence.
    """
    arr = labels[list(CRITERIA)].to_numpy()
    valid = (arr >= 0).all(axis=1)
    out = np.full(len(labels), -1, dtype=int)
    if not valid.any():
        return out
    tuples, inv = np.unique(arr[valid], axis=0, return_inverse=True)
    counts = np.bincount(inv, minlength=len(tuples))
    keep = counts >= min_cluster
    remap = np.where(keep, np.cumsum(keep) - 1, -1)
    out[valid] = remap[inv]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=2.0,
                    help="solar-longitude window width in degrees")
    ap.add_argument("--step", type=float, default=1.0,
                    help="window step in degrees; step < window gives overlap")
    ap.add_argument("--min-cluster", type=int, default=8)
    ap.add_argument("--fpr", type=float, default=0.001,
                    help="per-criterion false-positive rate used to calibrate thresholds")
    ap.add_argument("--max-window-size", type=int, default=6000,
                    help="subsample windows larger than this to keep the pairwise "
                         "matrix in memory; the subsample fraction is recorded")
    args = ap.parse_args()

    path = ROOT / "data" / "gmn_orbits.parquet"
    if not path.exists():
        print(f"missing {path}; run src/fetch_gmn.py first", file=sys.stderr)
        return 1
    df = pd.read_parquet(path)
    print(f"{len(df):,} meteors, {df.iau_code.ne('...').sum():,} with a GMN shower code")

    rng = np.random.default_rng(SEED)
    records, subsampled = [], 0
    starts = np.arange(0.0, 360.0, args.step)

    for w, start in enumerate(starts):
        lo, hi = start, start + args.window
        sel = ((df.sol_lon >= lo) & (df.sol_lon < hi)) if hi <= 360.0 else \
              ((df.sol_lon >= lo) | (df.sol_lon < hi - 360.0))
        win = df[sel]
        if len(win) < args.min_cluster * 3:
            continue
        frac = 1.0
        if len(win) > args.max_window_size:
            frac = args.max_window_size / len(win)
            win = win.sample(args.max_window_size, random_state=SEED)
            subsampled += 1

        thresholds = {c: sporadic_null_threshold(win, c, args.fpr, rng)
                      for c in CRITERIA}
        labels = cluster_window(win, thresholds, args.min_cluster)
        cons = consensus_groups(labels, args.min_cluster)

        for gid in np.unique(cons[cons >= 0]):
            members = win[cons == gid]
            codes = members.iau_code.value_counts()
            known = codes.index[0] if len(codes) and codes.index[0] != "..." else None
            known_frac = float(codes.iloc[0] / len(members)) if known else 0.0
            records.append({
                "window_start": float(start), "window_end": float(hi),
                "n_members": int(len(members)),
                "subsample_fraction": round(frac, 4),
                "dominant_iau": known, "dominant_iau_fraction": round(known_frac, 3),
                "sporadic_fraction": round(float((members.iau_code == "...").mean()), 3),
                "q_med": round(float(members.q.median()), 4),
                "e_med": round(float(members.e.median()), 4),
                "i_med": round(float(members.i.median()), 3),
                "peri_med": round(float(members.peri.median()), 3),
                "node_med": round(float(members.node.median()), 3),
                "vgeo_med": round(float(members.vgeo.median()), 3),
                "thr_d_sh": round(thresholds["d_sh"], 5),
                "thr_d_d": round(thresholds["d_d"], 5),
                "thr_d_h": round(thresholds["d_h"], 5),
            })
        if w % 30 == 0:
            print(f"  window {start:6.1f} deg: {len(win):5d} meteors, "
                  f"{len(np.unique(cons[cons >= 0]))} consensus groups")

    groups = pd.DataFrame(records)
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    groups.to_csv(out / "consensus_groups.csv", index=False)

    recovered = groups[groups.dominant_iau.notna()]
    novel = groups[groups.dominant_iau.isna()]
    summary = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "seed": SEED,
        "window_deg": args.window, "step_deg": args.step,
        "min_cluster": args.min_cluster, "fpr": args.fpr,
        "n_meteors": int(len(df)),
        "windows_subsampled": subsampled,
        "n_consensus_groups": int(len(groups)),
        "n_matching_known_shower": int(len(recovered)),
        "n_unmatched": int(len(novel)),
        "distinct_known_showers_recovered": int(recovered.dominant_iau.nunique())
        if len(recovered) else 0,
        "top_recovered": recovered.dominant_iau.value_counts().head(15).to_dict()
        if len(recovered) else {},
    }
    (out / "consensus_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("n_consensus_groups", "n_matching_known_shower", "n_unmatched",
                       "distinct_known_showers_recovered")}, indent=2))
    if len(recovered):
        print("recovered showers:", dict(list(
            recovered.dominant_iau.value_counts().head(10).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
