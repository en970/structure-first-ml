"""Filter the surviving structures by the one thing a chance alignment cannot fake.

The consensus sweep with significance testing cut the unmatched count from 3,727 to 498.
That is a large improvement and still not a candidate list. Two things are wrong with
reading 498 as discoveries:

  - Most of them sit in the degenerate regime documented in `density_excess`, where the
    null puts nothing in the group's ball, z collapses to the member count, and the
    statistic no longer measures contrast. Their z values are literally integers.
  - A few hundred genuine new showers per archive is not plausible. Real discovery rates
    are a handful per year across the whole field.

This module applies the physical filter that chance alignments cannot survive:
**multi-apparition recurrence**.

A meteoroid stream is a debris trail on a fixed heliocentric orbit. Earth crosses it at the
same solar longitude every year, so a real stream deposits meteors in the same narrow
orbital neighbourhood in year after year. A chance alignment in the sporadic background has
no such obligation: its members are drawn from whatever happened to be observed, and the
archive spans eight years, so a spurious clump concentrates in whichever year the
background happened to be dense.

WHAT THE FIRST VERSION OF THIS TEST GOT WRONG.

The obvious implementation is to gather every meteor within the candidate's orbital radius
and count how many distinct years contribute. That was tried, and it separated nothing:
99.5% of structures matching known showers recurred in three or more years, against 97.2%
of unmatched ones. The reason is that the sporadic background is present in that orbital
neighbourhood every year too, so "meteors appear here annually" carries no information at
all -- it is true of essentially every region of orbit space that Earth samples.

The correct question is not whether meteors recur but whether an EXCESS recurs. For each
apparition year separately, the count inside the candidate's radius is compared with what
the sporadic null puts there given that year's observing effort, which varied by a large
factor as the GMN camera network grew. A real stream produces an excess in year after
year; a chance alignment produces one in the year whose background happened to be dense
and nothing in the others.

This also corrects a scope limitation of the sweep, which only ever saw one
solar-longitude window at a time.

Run:  python3 src/analyze_candidates.py [--radius 0.12] [--min-years 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from consensus_cluster import physical_null  # noqa: E402
from dcriteria import d_sh  # noqa: E402
from iau_reference import load_iau_lists, match_to_iau  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SOL_LON_PAD = 5.0  # degrees either side when gathering an apparition


def jd_to_year(jd: np.ndarray) -> np.ndarray:
    """Calendar year from Julian date, good to well within a day."""
    return np.floor((jd - 2451545.0) / 365.25 + 2000.0).astype(int)


def window_of(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """All meteors near this candidate's solar longitude, across every year."""
    lo = (row.sol_lon - SOL_LON_PAD) % 360.0
    hi = (row.sol_lon + SOL_LON_PAD) % 360.0
    sel = ((df.sol_lon >= lo) & (df.sol_lon <= hi)) if lo <= hi else \
          ((df.sol_lon >= lo) | (df.sol_lon <= hi))
    return df[sel]


def distances_to(frame: pd.DataFrame, row: pd.Series) -> np.ndarray:
    return d_sh(frame.q.to_numpy(), frame.e.to_numpy(), frame.i.to_numpy(),
                frame.node.to_numpy(), frame.peri.to_numpy(),
                np.array([row.q]), np.array([row.e]), np.array([row.i]),
                np.array([row.node]), np.array([row.peri])).ravel()


def per_year_excess(win: pd.DataFrame, row: pd.Series, radius: float,
                    rng: np.random.Generator) -> tuple[dict[int, float], pd.DataFrame]:
    """Density-excess z inside the candidate radius, computed year by year.

    The null fraction is estimated once from the whole window and then scaled by each
    year's own meteor count, so a year in which the network recorded ten times as much
    is held to a ten times larger expectation rather than credited for it.
    """
    cols = ["q", "e", "i", "node", "peri"]
    null_df = physical_null(win[cols], rng, size=min(2000, len(win)))
    frac_null = float((distances_to(null_df, row) <= radius).mean())

    near = win[distances_to(win, row) <= radius]
    z_by_year: dict[int, float] = {}
    for year, n_year in win.groupby("year").size().items():
        n_obs = int((near.year == year).sum())
        n_exp = frac_null * n_year
        z_by_year[int(year)] = (n_obs - n_exp) / np.sqrt(max(n_exp, 1.0))
    return z_by_year, near


# Sporadic-source centres in sun-centred ecliptic coordinates (lambda - lambda_sun, beta),
# degrees. These are real concentrations in the sporadic complex and are NOT streams; a
# candidate sitting on one is almost certainly sporadic structure that an element-shuffled
# null cannot model, because the null destroys exactly the element correlations that
# produce them.
SPORADIC_SOURCES = {
    "helion": (340.0, 0.0),
    "antihelion": (200.0, 0.0),
    "north_apex": (270.0, 15.0),
    "south_apex": (270.0, -15.0),
    "north_toroidal": (270.0, 60.0),
}


def sporadic_source_distance(lam_sun_centred: float, beta: float) -> tuple[str, float]:
    """Nearest sporadic source and angular distance to it, in degrees."""
    best, best_d = None, np.inf
    for name, (sl, sb) in SPORADIC_SOURCES.items():
        dl = abs((lam_sun_centred - sl + 180.0) % 360.0 - 180.0)
        d = float(np.hypot(dl * np.cos(np.radians(0.5 * (beta + sb))), beta - sb))
        if d < best_d:
            best, best_d = name, d
    return best, best_d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=0.12,
                    help="orbital radius in D_SH for gathering an apparition")
    ap.add_argument("--min-years", type=int, default=3,
                    help="distinct apparition years required of a stream candidate")
    ap.add_argument("--min-per-year", type=int, default=2,
                    help="meteors needed in a year for it to count as present")
    ap.add_argument("--z-year", type=float, default=2.0,
                    help="per-year density-excess z required to call that year an "
                         "apparition of the stream rather than background")
    args = ap.parse_args()

    struct_path = ROOT / "outputs" / "consensus_structures.csv"
    if not struct_path.exists():
        print(f"missing {struct_path}; run src/consensus_cluster.py first", file=sys.stderr)
        return 1
    structures = pd.read_csv(struct_path)
    df = pd.read_parquet(ROOT / "data" / "gmn_orbits.parquet")
    df = df.assign(year=jd_to_year(df.jd.to_numpy()))
    years_available = sorted(df.year.unique())
    print(f"{len(structures)} structures; archive spans {years_available[0]}"
          f"-{years_available[-1]} ({len(years_available)} apparition years)")

    iau = load_iau_lists(ROOT / "data" / "iau")

    rng = np.random.default_rng(20260727)
    rows = []
    for _, row in structures.iterrows():
        win = window_of(df, row)
        if len(win) < 50:
            continue
        z_by_year, near = per_year_excess(win, row, args.radius, rng)
        if near.empty:
            continue
        excess_years = sorted(y for y, z in z_by_year.items() if z >= args.z_year)
        by_year = near.groupby("year").size()
        lam_sc = float((near.lamgeo.median() - near.sol_lon.median()) % 360.0)
        beta = float(near.betgeo.median())
        src, src_d = sporadic_source_distance(lam_sc, beta)
        rows.append({
            **row.to_dict(),
            "gathered": int(len(near)),
            "n_years_present": int((by_year >= args.min_per_year).sum()),
            "n_years_excess": len(excess_years),
            "excess_years": ",".join(str(y) for y in excess_years),
            "median_year_z": round(float(np.median(list(z_by_year.values()))), 2),
            "max_year_fraction": round(float(by_year.max() / by_year.sum()), 3),
            "gathered_sporadic_fraction": round(
                float((near.iau_code == "...").mean()), 3),
            "lam_sun_centred": round(lam_sc, 2), "beta_ecl": round(beta, 2),
            "nearest_sporadic_source": src,
            "sporadic_source_dist_deg": round(src_d, 1),
        })
    out = pd.DataFrame(rows)

    known = out.iau_established.notna() | out.iau_working.notna() | out.gmn_code.notna()
    recurring = out.n_years_excess >= args.min_years

    # Control: do the KNOWN showers pass the recurrence test? If they do not, the test is
    # wrong rather than the candidates. This is the whole justification for applying it.
    known_pass = float(recurring[known].mean()) if known.any() else float("nan")
    unknown_pass = float(recurring[~known].mean()) if (~known).any() else float("nan")
    print(f"\nrecurrence control: {known_pass:.1%} of matched-to-known structures show a "
          f"density excess in >= {args.min_years} years, against {unknown_pass:.1%} of "
          f"unmatched ones")
    if np.isfinite(known_pass) and np.isfinite(unknown_pass) and \
            known_pass - unknown_pass < 0.15:
        print("  WARNING: the test barely separates the two populations. Treat any "
              "candidate list below as provisional -- a filter that passes knowns and "
              "unknowns at the same rate is not filtering.")

    # Diagnostic: are the unmatched structures sitting on the sporadic sources?
    for label, mask in (("matched-to-known", known), ("unmatched", ~known)):
        sub = out[mask]
        if len(sub):
            on_source = float((sub.sporadic_source_dist_deg <= 25.0).mean())
            print(f"  {label:18s} {on_source:5.1%} lie within 25 deg of a sporadic "
                  f"source (median {sub.sporadic_source_dist_deg.median():.0f} deg)")

    candidates = out[(~known) & recurring].sort_values("n_years_excess", ascending=False)
    print(f"\n{int((~known).sum())} unmatched structures -> "
          f"{len(candidates)} survive multi-apparition recurrence")

    # Re-check survivors against both IAU lists at the gathered (larger) sample's centre,
    # since gathering across the full archive can shift the radiant slightly.
    final = []
    for _, c in candidates.iterrows():
        m = match_to_iau(c.ra, c.dec, c.vgeo, c.sol_lon, iau)
        if m["established"] or m["working"]:
            continue
        final.append({**c.to_dict(),
                      "iau_recheck_established": None, "iau_recheck_working": None})
    final_df = pd.DataFrame(final)

    outputs = ROOT / "outputs"
    out.to_csv(outputs / "structures_with_recurrence.csv", index=False)
    if len(final_df):
        final_df.to_csv(outputs / "candidates.csv", index=False)

    summary = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "radius_dsh": args.radius, "min_years": args.min_years,
        "min_per_year": args.min_per_year,
        "apparition_years": [int(years_available[0]), int(years_available[-1])],
        "z_year": args.z_year,
        "n_structures": int(len(out)),
        "n_known": int(known.sum()),
        "n_unmatched": int((~known).sum()),
        "known_recurrence_rate": round(known_pass, 4),
        "unmatched_recurrence_rate": round(unknown_pass, 4),
        "n_candidates_after_recurrence": int(len(candidates)),
        "n_candidates_after_iau_recheck": int(len(final_df)),
    }
    (outputs / "candidates_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if len(final_df):
        print("\nsurviving candidates (still NOT discoveries -- see README):")
        cols = ["n_members", "gathered", "n_years_excess", "excess_years",
                "median_year_z", "best_z", "ra", "dec", "vgeo", "sol_lon"]
        print(final_df.head(20)[cols].to_string(index=False))
    else:
        print("\nNo structure survives both recurrence and the IAU re-check. "
              "That is a null result and it is the honest headline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
