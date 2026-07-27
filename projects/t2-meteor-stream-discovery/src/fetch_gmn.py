"""Download and assemble the Global Meteor Network trajectory-summary archive.

The GMN publishes monthly trajectory summaries as semicolon-separated text files under a
CC BY 4.0 licence. Each row is one multi-station meteor with a full trajectory and orbit
solution: radiant and velocity with per-quantity uncertainties, osculating orbital
elements, Tisserand parameter, convergence angle, fit error, station count, and the GMN
pipeline's own IAU shower association. Access was verified live on 2026-07-27; the
December 2025 file alone holds 118,023 meteors in 86 columns.

The GMN shower code is retained as `iau_code` but is never used as an input to
clustering. It is the positive control: a method that cannot recover the Perseids has no
standing to propose anything new. Using the pipeline's own association as a feature would
make the exercise circular.

Quality cuts follow the spirit of the GMN team's own analyses (Vida et al.); the exact
thresholds are parameters, are recorded in the output summary, and the fraction removed
by each cut is reported rather than hidden.

Run:  python3 src/fetch_gmn.py [--start 201812] [--end 202606] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://globalmeteornetwork.org/data/traj_summary_data/monthly/"
ROOT = Path(__file__).resolve().parent.parent

# The 86 columns of the trajectory summary format, in file order. Names verified against
# the header of traj_summary_monthly_202512.txt on 2026-07-27.
COLUMNS = [
    "traj_id", "jd", "utc", "iau_no", "iau_code",
    "sol_lon", "app_lst",
    "rageo", "rageo_sig", "decgeo", "decgeo_sig",
    "lamgeo", "lamgeo_sig", "betgeo", "betgeo_sig",
    "vgeo", "vgeo_sig",
    "lamhel", "lamhel_sig", "bethel", "bethel_sig",
    "vhel", "vhel_sig",
    "a", "a_sig", "e", "e_sig", "i", "i_sig",
    "peri", "peri_sig", "node", "node_sig",
    "pi", "pi_sig", "b", "b_sig", "q", "q_sig",
    "f", "f_sig", "M", "M_sig", "Q", "Q_sig",
    "n", "n_sig", "T", "T_sig",
    "tisserand", "tisserand_sig",
    "raapp", "raapp_sig", "decapp", "decapp_sig",
    "azim", "azim_sig", "elev", "elev_sig",
    "vinit", "vinit_sig", "vavg", "vavg_sig",
    "latbeg", "latbeg_sig", "lonbeg", "lonbeg_sig", "htbeg", "htbeg_sig",
    "latend", "latend_sig", "lonend", "lonend_sig", "htend", "htend_sig",
    "duration", "peak_mag", "peak_ht", "F_param", "mass_kg",
    "qc", "median_fit_err", "beg_in_fov", "end_in_fov",
    "num_stations", "stations",
]

# Columns actually needed downstream. Everything else is dropped after parsing to keep
# the assembled archive small enough to hold in memory comfortably.
KEEP = [
    "traj_id", "jd", "iau_no", "iau_code", "sol_lon",
    "rageo", "rageo_sig", "decgeo", "decgeo_sig",
    "lamgeo", "betgeo", "vgeo", "vgeo_sig",
    "a", "e", "e_sig", "i", "i_sig", "peri", "peri_sig", "node", "node_sig",
    "q", "q_sig", "tisserand", "qc", "median_fit_err", "num_stations",
]

# Quality-cut defaults. Reported, not hidden; adjustable from the command line.
DEFAULT_CUTS = {
    "qc_min_deg": 15.0,        # minimum convergence angle
    "radiant_sig_max_deg": 1.0,  # combined radiant uncertainty
    "vgeo_sig_max_kms": 1.0,   # geocentric-velocity uncertainty
    "e_max": 1.2,              # generous hyperbolic margin; stricter cut happens later
    "q_min_au": 0.0,
}


def month_range(start: str, end: str) -> list[str]:
    months, cur = [], start
    while cur <= end:
        y, m = int(cur[:4]), int(cur[4:])
        months.append(cur)
        m += 1
        if m > 12:
            y, m = y + 1, 1
        cur = f"{y:04d}{m:02d}"
    return months


def fetch_month(month: str, cache: Path, retries: int = 3) -> Path | None:
    """Download one monthly file into the cache, resuming across runs."""
    out = cache / f"traj_summary_monthly_{month}.txt"
    if out.exists() and out.stat().st_size > 0:
        return out
    url = f"{BASE}traj_summary_monthly_{month}.txt"
    for attempt in range(retries):
        try:
            with requests.get(url, timeout=600, stream=True) as resp:
                if resp.status_code == 404:
                    return None  # month not published
                resp.raise_for_status()
                tmp = out.with_suffix(".part")
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                tmp.rename(out)
                return out
        except requests.RequestException:
            time.sleep(5.0 * (attempt + 1))
    return None


def parse_month(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", comment="#", header=None, names=COLUMNS,
                     skipinitialspace=True, low_memory=False)
    df = df[KEEP].copy()
    df["iau_code"] = df["iau_code"].astype(str).str.strip()
    for col in KEEP:
        if col not in ("traj_id", "iau_code"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def apply_cuts(df: pd.DataFrame, cuts: dict) -> tuple[pd.DataFrame, dict]:
    """Sequential quality cuts, each with its removal count recorded."""
    report, n0 = {}, len(df)

    steps = [
        ("missing_orbit", df[["a", "e", "i", "peri", "node", "q"]].notna().all(axis=1)),
        ("qc", df["qc"] >= cuts["qc_min_deg"]),
        ("radiant_sig", np.hypot(df["rageo_sig"], df["decgeo_sig"])
         <= cuts["radiant_sig_max_deg"]),
        ("vgeo_sig", df["vgeo_sig"] <= cuts["vgeo_sig_max_kms"]),
        ("eccentricity", df["e"] <= cuts["e_max"]),
        ("perihelion", df["q"] > cuts["q_min_au"]),
    ]
    mask = pd.Series(True, index=df.index)
    for name, cond in steps:
        cond = cond.fillna(False)
        removed = int((mask & ~cond).sum())
        report[f"removed_{name}"] = removed
        mask &= cond

    out = df[mask].reset_index(drop=True)
    report["rows_before"] = n0
    report["rows_after"] = int(len(out))
    return out, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="201812")
    ap.add_argument("--end", default="202606")
    ap.add_argument("--workers", type=int, default=4)
    for key, val in DEFAULT_CUTS.items():
        ap.add_argument(f"--{key.replace('_', '-')}", type=float, default=val)
    args = ap.parse_args()
    cuts = {k: getattr(args, k) for k in DEFAULT_CUTS}

    cache = ROOT / "data" / "gmn_monthly"
    cache.mkdir(parents=True, exist_ok=True)
    months = month_range(args.start, args.end)
    print(f"{len(months)} months requested ({args.start}..{args.end})")

    paths: list[Path] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_month, m, cache): m for m in months}
        for fut in as_completed(futures):
            p = fut.result()
            if p is not None:
                paths.append(p)
    print(f"downloaded/cached {len(paths)} files in {time.time() - t0:.0f} s "
          f"({sum(p.stat().st_size for p in paths) / 1e9:.2f} GB)")

    frames = []
    for p in sorted(paths):
        frames.append(parse_month(p))
    df = pd.concat(frames, ignore_index=True)
    print(f"parsed {len(df):,} meteors")

    df, report = apply_cuts(df, cuts)
    print(f"after quality cuts: {len(df):,} meteors")
    for k, v in report.items():
        if k.startswith("removed_"):
            print(f"  {k}: {v:,}")

    out = ROOT / "data" / "gmn_orbits.parquet"
    df.to_parquet(out, index=False)

    shower_counts = (df[df.iau_code != "..."].groupby("iau_code").size()
                     .sort_values(ascending=False))
    summary = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "months": [args.start, args.end],
        "files": len(paths),
        "cuts": cuts,
        "cut_report": report,
        "n_meteors": int(len(df)),
        "n_sporadic": int((df.iau_code == "...").sum()),
        "n_shower_associated": int((df.iau_code != "...").sum()),
        "n_distinct_showers": int(shower_counts.size),
        "top_showers": shower_counts.head(15).to_dict(),
    }
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    (outputs / "fetch_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("n_meteors", "n_sporadic", "n_shower_associated",
                       "n_distinct_showers")}, indent=2))
    print("top showers:", dict(list(shower_counts.head(8).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
