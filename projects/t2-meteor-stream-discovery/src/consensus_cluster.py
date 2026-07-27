"""Consensus clustering across orbital dissimilarity criteria, with significance testing.

The method the field's own review asks for and nobody has run on this archive: cluster the
orbits independently under several classical D-criteria, keep only structure that survives
all of them, and then test what survives against a physically valid sporadic null.

WHAT THE FIRST VERSION GOT WRONG, AND WHY THIS ONE EXISTS.

A first pass ran consensus clustering with per-criterion thresholds calibrated to a common
false-positive rate, and returned 4,517 groups of which 3,727 matched no known shower.
Read naively that is a spectacular result. It is nothing of the kind, and the audit that
caught it is worth recording:

  - The median unmatched group had a sporadic fraction of 1.000 and 14 members, while the
    largest had 5,804 members in a 6,000-meteor window. DBSCAN chains through a dense,
    continuous background: no matter how small the neighbourhood radius, sufficient
    density links everything into one component. Those "clusters" were the sporadic
    complex itself, not streams in it.
  - Overlapping windows counted the same structure repeatedly. 790 matched groups
    corresponded to only 169 distinct showers, so a typical stream was counted about five
    times; the unmatched count was inflated the same way.
  - "Unmatched" was decided by whether the single most common code in a group was the
    sporadic marker. A genuine stream whose members GMN mostly left unlabelled was
    therefore filed as novel.

Four changes follow from that, and they are the substance of this module.

1. PHYSICALLY VALID NULL. Shuffling orbital elements independently preserves every
   marginal but produces orbits that cannot reach Earth. A meteoroid is only observable if
   its orbit crosses Earth's, so null orbits failing q <= Q_earth and Q >= q_earth are
   rejected and redrawn. Without this the null is dispersed over unreachable orbital space,
   its pair distances are inflated, and every threshold derived from it is too permissive.

2. GROUP-LEVEL SIGNIFICANCE. Each surviving group is tested as an object, not assumed
   real because a clustering algorithm emitted it. The statistic is a density excess: how
   many meteors fall within the group's own orbital radius, against how many the null puts
   there. A chained background component covers a large volume at background density and
   scores near zero however many members it has.

3. DEDUPLICATION ACROSS WINDOWS. Groups sharing members are merged by connected
   components, so a stream active over ten degrees of solar longitude is reported once.

4. BOTH IAU LISTS. Novelty is checked against the established list AND the working list of
   787 candidate showers. Checking only the established list would manufacture novelty out
   of showers someone has already reported.

Run:  python3 src/consensus_cluster.py [--window 2.0] [--step 1.0] [--min-cluster 10]
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
from iau_reference import load_iau_lists, match_to_iau  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260727
CRITERIA = ("d_sh", "d_d", "d_h")
FN = {"d_sh": d_sh, "d_d": d_d, "d_h": d_h}
SPORADIC = "..."

# Earth's orbit, for the crossing constraint on null orbits.
Q_EARTH, A_EARTH = 0.9833, 1.0167


def orbit_cols(df: pd.DataFrame) -> tuple:
    return (df["q"].to_numpy(), df["e"].to_numpy(), df["i"].to_numpy(),
            df["node"].to_numpy(), df["peri"].to_numpy())


def pairwise(df: pd.DataFrame, criterion: str) -> np.ndarray:
    q, e, i, node, peri = orbit_cols(df)
    return FN[criterion](q, e, i, node, peri, q, e, i, node, peri)


def earth_crossing(q: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Can this orbit reach Earth? q <= Earth aphelion and aphelion >= Earth perihelion."""
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = np.where(e < 1.0, q * (1.0 + e) / np.maximum(1.0 - e, 1e-12), np.inf)
    return (q <= A_EARTH) & (Q >= Q_EARTH)


def physical_null(df: pd.DataFrame, rng: np.random.Generator,
                  size: int, max_tries: int = 12) -> pd.DataFrame:
    """Element-wise shuffle, restricted to orbits that can actually reach Earth.

    Preserves each element's marginal distribution -- so the helion and antihelion
    concentrations, the eccentricity and inclination distributions all survive -- while
    destroying the joint structure that constitutes a stream.
    """
    cols = ("q", "e", "i", "node", "peri")
    kept: list[pd.DataFrame] = []
    n_have = 0
    for _ in range(max_tries):
        draw = pd.DataFrame({c: rng.permutation(df[c].to_numpy()) for c in cols})
        ok = draw[earth_crossing(draw.q.to_numpy(), draw.e.to_numpy())]
        if len(ok):
            kept.append(ok)
            n_have += len(ok)
        if n_have >= size:
            break
    if not kept:
        return df[list(cols)].sample(min(size, len(df)), random_state=SEED)
    out = pd.concat(kept, ignore_index=True)
    return out.iloc[:size] if len(out) > size else out


def calibrate_threshold(null_df: pd.DataFrame, criterion: str, fpr: float) -> float:
    """Distance threshold admitting a fixed fraction of chance pairs from the null.

    The quantile is taken over the full upper triangle of the null pairwise matrix rather
    than a sample of pairs. A first version drew 3,000 sampled pairs, which put the
    0.001 quantile at the third order statistic -- so noisy that neighbouring
    solar-longitude windows produced thresholds differing by an order of magnitude. With
    2,500 null orbits the triangle holds about 3.1 million distances and the same
    quantile rests on roughly 3,000 of them.
    """
    n = len(null_df)
    if n < 50:
        return float("nan")
    d = pairwise(null_df, criterion)
    iu = np.triu_indices(n, k=1)
    return float(np.quantile(d[iu], fpr))


def density_excess(members: pd.DataFrame, window: pd.DataFrame,
                   null_df: pd.DataFrame, criterion: str = "d_sh") -> dict:
    """Significance of a group as a density excess over the sporadic null.

    The group's centroid orbit and its own radius (the 80th percentile of member
    distances from that centroid) define a ball in orbit space. The observed count in
    that ball is compared with the count the null puts in the same ball, rescaled to the
    window's size. A chained background component has a huge radius and sits at background
    density, so it scores near zero regardless of member count.
    """
    cen = {c: float(np.median(members[c])) for c in ("q", "e", "i", "node", "peri")}
    fn = FN[criterion]

    def to_centre(frame: pd.DataFrame) -> np.ndarray:
        q, e, i, node, peri = orbit_cols(frame)
        return fn(q, e, i, node, peri,
                  np.array([cen["q"]]), np.array([cen["e"]]), np.array([cen["i"]]),
                  np.array([cen["node"]]), np.array([cen["peri"]])).ravel()

    d_mem = to_centre(members)
    radius = float(np.quantile(d_mem, 0.8))
    if not np.isfinite(radius) or radius <= 0:
        return {"radius": radius, "n_obs": len(members), "n_exp": np.nan, "z": np.nan}

    n_obs = int((to_centre(window) <= radius).sum())
    d_null = to_centre(null_df)
    frac_null = float((d_null <= radius).mean())
    n_exp = frac_null * len(window)
    z = (n_obs - n_exp) / np.sqrt(max(n_exp, 1.0))
    return {"radius": round(radius, 5), "n_obs": n_obs,
            "n_exp": round(float(n_exp), 2), "z": round(float(z), 2)}


def consensus_labels(df: pd.DataFrame, thresholds: dict, min_cluster: int) -> np.ndarray:
    """Meteors grouped only when every criterion places them in the same cluster."""
    per = {}
    for crit in CRITERIA:
        eps = thresholds[crit]
        if not np.isfinite(eps):
            return np.full(len(df), -1, dtype=int)
        per[crit] = DBSCAN(eps=eps, min_samples=min_cluster,
                           metric="precomputed").fit_predict(pairwise(df, crit))
    arr = np.column_stack([per[c] for c in CRITERIA])
    valid = (arr >= 0).all(axis=1)
    out = np.full(len(df), -1, dtype=int)
    if not valid.any():
        return out
    tuples, inv = np.unique(arr[valid], axis=0, return_inverse=True)
    counts = np.bincount(inv, minlength=len(tuples))
    keep = counts >= min_cluster
    remap = np.where(keep, np.cumsum(keep) - 1, -1)
    out[valid] = remap[inv]
    return out


def merge_overlapping(groups: list[dict]) -> list[dict]:
    """Merge groups sharing members, by connected components over the member sets."""
    parent = list(range(len(groups)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    seen: dict[str, int] = {}
    for gi, g in enumerate(groups):
        for mid in g["members"]:
            if mid in seen:
                union(seen[mid], gi)
            else:
                seen[mid] = gi

    buckets: dict[int, list[int]] = {}
    for gi in range(len(groups)):
        buckets.setdefault(find(gi), []).append(gi)

    merged = []
    for members_of in buckets.values():
        parts = [groups[i] for i in members_of]
        ids: set[str] = set()
        for p in parts:
            ids |= p["members"]
        best = max(parts, key=lambda p: p["significance"]["z"]
                   if np.isfinite(p["significance"]["z"]) else -np.inf)
        merged.append({
            "members": ids,
            "n_windows": len(parts),
            "n_members": len(ids),
            "sol_lon_min": min(p["window_start"] for p in parts),
            "sol_lon_max": max(p["window_end"] for p in parts),
            "best_z": best["significance"]["z"],
            "representative": best,
        })
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--min-cluster", type=int, default=10)
    ap.add_argument("--fpr", type=float, default=0.001)
    ap.add_argument("--max-window-size", type=int, default=6000)
    ap.add_argument("--z-min", type=float, default=5.0,
                    help="minimum density-excess z for a group to be reported at all")
    ap.add_argument("--max-radius", type=float, default=0.30,
                    help="maximum group radius in D_SH. A meteoroid stream is compact: "
                         "classical association thresholds sit at 0.05-0.2, so a group "
                         "whose 80th-percentile member distance exceeds this is a chained "
                         "background component, not a stream. Without this ceiling a "
                         "4,844-member group of radius 1.98 passed at z=11.2.")
    args = ap.parse_args()

    path = ROOT / "data" / "gmn_orbits.parquet"
    if not path.exists():
        print(f"missing {path}; run src/fetch_gmn.py first", file=sys.stderr)
        return 1
    df = pd.read_parquet(path)
    print(f"{len(df):,} meteors; {df.iau_code.ne(SPORADIC).sum():,} carry a GMN code")

    iau = load_iau_lists(ROOT / "data" / "iau")
    print(f"IAU reference: {iau['established'].code.nunique()} established codes, "
          f"{iau['working'].code.nunique()} working-list codes")

    rng = np.random.default_rng(SEED)
    raw_groups: list[dict] = []
    subsampled = 0

    for start in np.arange(0.0, 360.0, args.step):
        lo, hi = start, start + args.window
        sel = ((df.sol_lon >= lo) & (df.sol_lon < hi)) if hi <= 360.0 else \
              ((df.sol_lon >= lo) | (df.sol_lon < hi - 360.0))
        win = df[sel]
        if len(win) < args.min_cluster * 4:
            continue
        if len(win) > args.max_window_size:
            win = win.sample(args.max_window_size, random_state=SEED)
            subsampled += 1

        null_df = physical_null(win, rng, size=min(2500, len(win)))
        thresholds = {c: calibrate_threshold(null_df, c, args.fpr) for c in CRITERIA}
        labels = consensus_labels(win, thresholds, args.min_cluster)

        for gid in np.unique(labels[labels >= 0]):
            members = win[labels == gid]
            sig = density_excess(members, win, null_df)
            if not np.isfinite(sig["z"]) or sig["z"] < args.z_min:
                continue
            if not np.isfinite(sig["radius"]) or sig["radius"] > args.max_radius:
                continue  # chained background, not a stream
            codes = members.iau_code.value_counts()
            named = codes[codes.index != SPORADIC]
            raw_groups.append({
                "members": set(members.traj_id),
                "window_start": float(start), "window_end": float(hi),
                "significance": sig,
                "gmn_code": (named.index[0] if len(named) else None),
                "gmn_code_fraction": (round(float(named.iloc[0] / len(members)), 3)
                                      if len(named) else 0.0),
                "sporadic_fraction": round(float((members.iau_code == SPORADIC).mean()), 3),
                "ra": float(members.rageo.median()), "dec": float(members.decgeo.median()),
                "vgeo": float(members.vgeo.median()),
                "sol_lon": float(members.sol_lon.median()),
                "q": float(members.q.median()), "e": float(members.e.median()),
                "i": float(members.i.median()), "peri": float(members.peri.median()),
                "node": float(members.node.median()),
            })

    print(f"{len(raw_groups)} window-level groups passed z >= {args.z_min} "
          f"({subsampled} windows subsampled)")

    merged = merge_overlapping(raw_groups)
    print(f"{len(merged)} distinct structures after merging overlapping windows")

    rows = []
    for m in merged:
        r = m["representative"]
        match = match_to_iau(r["ra"], r["dec"], r["vgeo"], r["sol_lon"], iau)
        rows.append({
            "n_members": m["n_members"], "n_windows": m["n_windows"],
            "sol_lon_min": m["sol_lon_min"], "sol_lon_max": m["sol_lon_max"],
            "best_z": m["best_z"],
            "ra": round(r["ra"], 2), "dec": round(r["dec"], 2),
            "vgeo": round(r["vgeo"], 2), "sol_lon": round(r["sol_lon"], 2),
            "q": round(r["q"], 4), "e": round(r["e"], 4), "i": round(r["i"], 2),
            "peri": round(r["peri"], 2), "node": round(r["node"], 2),
            "gmn_code": r["gmn_code"], "gmn_code_fraction": r["gmn_code_fraction"],
            "sporadic_fraction": r["sporadic_fraction"],
            "iau_established": (match["established"]["code"] if match["established"] else None),
            "iau_established_sep": (match["established"]["separation_deg"]
                                    if match["established"] else None),
            "iau_working": (match["working"]["code"] if match["working"] else None),
            "iau_working_sep": (match["working"]["separation_deg"]
                                if match["working"] else None),
        })
    out_df = pd.DataFrame(rows).sort_values("best_z", ascending=False)

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    out_df.to_csv(out / "consensus_structures.csv", index=False)

    known_any = out_df.iau_established.notna() | out_df.iau_working.notna() | \
        out_df.gmn_code.notna()
    unmatched = out_df[~known_any]
    summary = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "seed": SEED, "window_deg": args.window, "step_deg": args.step,
        "min_cluster": args.min_cluster, "fpr": args.fpr, "z_min": args.z_min,
        "max_radius": args.max_radius,
        "n_meteors": int(len(df)), "windows_subsampled": subsampled,
        "n_window_groups": len(raw_groups),
        "n_distinct_structures": int(len(out_df)),
        "n_matched_established": int(out_df.iau_established.notna().sum()),
        "n_matched_working_only": int((out_df.iau_working.notna()
                                       & out_df.iau_established.isna()).sum()),
        "n_gmn_code_only": int((out_df.gmn_code.notna()
                                & out_df.iau_established.isna()
                                & out_df.iau_working.isna()).sum()),
        "n_unmatched": int(len(unmatched)),
        "distinct_established_recovered": int(out_df.iau_established.nunique()),
        "unmatched_median_z": (round(float(unmatched.best_z.median()), 2)
                               if len(unmatched) else None),
        "unmatched_median_members": (int(unmatched.n_members.median())
                                     if len(unmatched) else None),
    }
    (out / "consensus_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("n_distinct_structures", "n_matched_established",
                       "n_matched_working_only", "n_gmn_code_only", "n_unmatched",
                       "distinct_established_recovered")}, indent=2))
    if len(unmatched):
        print("\ntop unmatched by significance (candidates, NOT discoveries):")
        print(unmatched.head(10)[["n_members", "best_z", "ra", "dec", "vgeo",
                                  "sol_lon", "sporadic_fraction"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
