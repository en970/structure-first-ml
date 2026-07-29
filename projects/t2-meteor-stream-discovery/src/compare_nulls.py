"""Measure the three sporadic-background models against what a null has to do.

A null model for stream detection has two jobs, and they pull in opposite directions:

  CALIBRATION   In a region of orbit space containing only sporadic meteors, the null must
                predict roughly the number actually observed. A null that under-predicts
                everywhere manufactures significance; one that over-predicts hides real
                streams. The statistic is the ratio n_exp / n_obs, which should sit near 1.

  DISCRIMINATION  In a region containing a real stream, the null must predict far fewer
                meteors than are observed -- that gap is the detection. The same ratio
                should be far below 1.

The failure diagnosed in RESULTS.md is a calibration failure: the permutation null
under-predicts in sporadic regions because it destroys the element correlations that
create the sporadic sources, so ordinary background structure scores as excess. This
module tests that diagnosis directly instead of assuming it, and measures whether the KDE
model fixes it.

Sporadic test regions are centred on meteors GMN labels sporadic; the stream test regions
are centred on the strongest showers active in each window, which GMN labels independently
of anything computed here.

Run:  python3 src/compare_nulls.py [--n-sporadic 25] [--windows 140 195 260]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dcriteria import d_sh  # noqa: E402
from sporadic_null import kde_null, permutation_null, sideband_null  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260729
SPORADIC = "..."
RADIUS = 0.10  # D_SH ball used for both kinds of test region


def dist_to(frame: pd.DataFrame, centre: pd.Series) -> np.ndarray:
    return d_sh(frame.q.to_numpy(), frame.e.to_numpy(), frame.i.to_numpy(),
                frame.node.to_numpy(), frame.peri.to_numpy(),
                np.array([centre.q]), np.array([centre.e]), np.array([centre.i]),
                np.array([centre.node]), np.array([centre.peri])).ravel()


def ratio(centre: pd.Series, window: pd.DataFrame, null: pd.DataFrame) -> tuple:
    """n_exp / n_obs inside the ball, plus the raw counts."""
    n_obs = int((dist_to(window, centre) <= RADIUS).sum())
    frac = float((dist_to(null, centre) <= RADIUS).mean())
    n_exp = frac * len(window)
    return (n_exp / max(n_obs, 1)), n_obs, n_exp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sporadic", type=int, default=25)
    ap.add_argument("--windows", type=float, nargs="+", default=[140.0, 195.0, 260.0])
    ap.add_argument("--window-width", type=float, default=2.0)
    ap.add_argument("--null-size", type=int, default=2500)
    args = ap.parse_args()

    df = pd.read_parquet(ROOT / "data" / "gmn_orbits.parquet")
    rng = np.random.default_rng(SEED)
    records = []

    for centre_lon in args.windows:
        lo, hi = centre_lon, centre_lon + args.window_width
        win = df[(df.sol_lon >= lo) & (df.sol_lon < hi)]
        if len(win) > 6000:
            win = win.sample(6000, random_state=SEED)
        if len(win) < 200:
            continue

        nulls = {
            "permutation": permutation_null(win, rng, args.null_size),
            "kde": kde_null(win, rng, args.null_size),
            "sideband": sideband_null(df, centre_lon + args.window_width / 2, rng,
                                      args.null_size),
        }

        # Stream region: the most abundant GMN-labelled shower in this window.
        codes = win[win.iau_code != SPORADIC].iau_code.value_counts()
        stream_code = codes.index[0] if len(codes) else None
        stream_centre = None
        if stream_code is not None and codes.iloc[0] >= 30:
            members = win[win.iau_code == stream_code]
            stream_centre = pd.Series({c: float(members[c].median())
                                       for c in ("q", "e", "i", "node", "peri")})

        # Sporadic regions: centred on meteors GMN calls sporadic.
        spor = win[win.iau_code == SPORADIC]
        centres = spor.sample(min(args.n_sporadic, len(spor)), random_state=SEED)

        for name, null in nulls.items():
            spor_ratios = []
            for _, c in centres.iterrows():
                r, n_obs, _ = ratio(c, win, null)
                if n_obs >= 5:
                    spor_ratios.append(r)
            rec = {"window": centre_lon, "null": name,
                   "sporadic_median_ratio": (round(float(np.median(spor_ratios)), 3)
                                             if spor_ratios else None),
                   "n_sporadic_regions": len(spor_ratios)}
            if stream_centre is not None:
                r, n_obs, n_exp = ratio(stream_centre, win, null)
                rec.update({"stream_code": stream_code, "stream_ratio": round(r, 3),
                            "stream_n_obs": n_obs, "stream_n_exp": round(n_exp, 1)})
            records.append(rec)

    out = pd.DataFrame(records)
    print(out.to_string(index=False))

    print("\nsummary across windows (lower stream ratio = better discrimination, "
          "sporadic ratio near 1 = better calibration)")
    agg = out.groupby("null").agg(
        sporadic_ratio=("sporadic_median_ratio", "median"),
        stream_ratio=("stream_ratio", "median")).round(3)
    agg["separation"] = (agg.sporadic_ratio / agg.stream_ratio).round(2)
    print(agg.to_string())

    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    out.to_csv(outputs / "null_comparison.csv", index=False)
    (outputs / "null_comparison.json").write_text(json.dumps({
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "seed": SEED, "radius_dsh": RADIUS, "windows": args.windows,
        "records": records,
        "summary": agg.reset_index().to_dict(orient="records"),
    }, indent=2))

    best = agg.separation.idxmax()
    print(f"\nbest separation: {best} ({agg.loc[best, 'separation']}x)")
    print("A null that is calibrated on sporadic regions (ratio near 1) while predicting "
          "far too few meteors at a real stream (ratio well below 1) is the one to use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
