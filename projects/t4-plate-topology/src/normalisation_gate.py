"""The gate: does plate-level normalisation make epoch comparison possible at all?

T4's falsification run left one test essentially flat. Diagrams of the SAME sky position on
different plates were 1,157 apart in bottleneck distance while diagrams of DIFFERENT fields
were 1,183 apart -- a separation of 2.2%, which passes the stated criterion and means
nothing in practice. Plate-to-plate variation in emulsion, exposure time, limiting magnitude
and development is comparable to the difference between one patch of sky and another.

That matters because the century-long time axis is the entire reason to work this archive.
If topology cannot tell the same field from a different one across plates, it certainly
cannot track how a field CHANGES across decades.

THIS IS A GATE, NOT AN IMPROVEMENT. If normalisation does not open the separation, the
temporal-topology idea fails and the track closes with that recorded.

Three normalisations, and one of them is principled rather than merely conventional:

  none        raw cutout pixels, the state that produced the 2.2% result

  zscore      (x - median) / MAD per patch. Removes additive and multiplicative differences
              in sky level and contrast, which is the obvious first thing to try.

  rank        replace each pixel by its rank within the patch, rescaled to [0, 1]. This is
              the principled one for THIS method: sublevel-set persistence depends only on
              the ORDER in which pixels enter the filtration, so any strictly monotone
              transform of intensity leaves the ordering intact and moves the diagram only
              along the filtration axis. Rank normalisation is the canonical representative
              of that equivalence class -- it maps every plate onto a common filtration
              scale while preserving exactly the information persistence actually uses.
              Emulsion response is monotone but not linear in exposure, so this removes a
              real nuisance rather than an imagined one.

If the rank transform does not open the gap, the failure is not calibration: it means the
topology of these plates is dominated by grain and defects whose ordering also differs
plate to plate, and no monotone correction can repair that.

Run:  python3 src/normalisation_gate.py [--n-plates 14] [--offset 0.5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dasch_access import (cutout, exposures, plate_epoch,  # noqa: E402
                          plate_metadata, search_plates)
from falsify_grain import (PATCH, bottleneck, diagram_summary,  # noqa: E402
                           persistence_diagram, source_fraction)

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260803


def norm_none(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, dtype=np.float64)


def norm_zscore(p: np.ndarray) -> np.ndarray:
    a = np.asarray(p, dtype=np.float64)
    med = np.nanmedian(a)
    mad = np.nanmedian(np.abs(a - med))
    return (a - med) / (mad if mad > 0 else 1.0)


def norm_rank(p: np.ndarray) -> np.ndarray:
    """Rank transform, rescaled to [0, 1].

    Sublevel-set persistence depends only on the order pixels enter the filtration, so this
    is the canonical representative of the monotone-transform equivalence class the method
    is already invariant to in shape but not in scale.
    """
    a = np.asarray(p, dtype=np.float64).ravel()
    order = np.argsort(np.argsort(a))
    return (order / max(len(a) - 1, 1)).reshape(p.shape)


NORMS = {"none": norm_none, "zscore": norm_zscore, "rank": norm_rank}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-plates", type=int, default=14)
    ap.add_argument("--offset", type=float, default=0.5,
                    help="degrees to offset the comparison field")
    ap.add_argument("--ra", type=float, default=11.446426)
    ap.add_argument("--dec", type=float, default=-71.535997)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    ids = search_plates(args.ra, args.dec)
    print(f"{len(ids)} plates cover the field")

    at_field, off_field, meta_rows = [], [], []
    for pid in ids:
        if len(at_field) >= args.n_plates:
            break
        meta = plate_metadata(pid)
        if meta is None:
            continue
        exps = exposures(meta)
        if not exps or exps[0].get("ctr_ra") is None:
            continue
        ex = exps[0]
        c = cutout(pid, float(ex["ctr_ra"]), float(ex["ctr_dec"]), 0)
        if c is None:
            continue
        c2 = cutout(pid, float(ex["ctr_ra"]) + args.offset, float(ex["ctr_dec"]), 0)
        if c2 is None:
            continue
        at_field.append((pid, c[0][:PATCH, :PATCH]))
        off_field.append((pid, c2[0][:PATCH, :PATCH]))
        meta_rows.append({
            "plate_id": pid, "epoch": plate_epoch(meta),
            "exposure_length": ex.get("exposure_length"),
            "source_fraction": round(source_fraction(c[0]), 5),
            "median": float(np.nanmedian(c[0])),
            "mad": float(np.nanmedian(np.abs(c[0] - np.nanmedian(c[0])))),
        })
        print(f"  {pid} {plate_epoch(meta)} exp={ex.get('exposure_length')}s "
              f"src_frac={meta_rows[-1]['source_fraction']:.5f} "
              f"mad={meta_rows[-1]['mad']:.1f}")

    if len(at_field) < 4:
        print("fewer than four usable plates; the gate cannot be evaluated",
              file=sys.stderr)
        return 1

    plates = pd.DataFrame(meta_rows)
    # How heterogeneous are these plates before any correction? This is the quantity
    # normalisation has to remove.
    het = {
        "n_plates": int(len(plates)),
        "epoch_range": [plates.epoch.min(), plates.epoch.max()],
        "mad_cv": round(float(plates.mad.std() / max(plates.mad.mean(), 1e-9)), 3),
        "median_cv": round(float(plates["median"].std()
                                 / max(plates["median"].mean(), 1e-9)), 3),
        "source_fraction_cv": round(
            float(plates.source_fraction.std()
                  / max(plates.source_fraction.mean(), 1e-9)), 3),
    }
    print(f"\nplate heterogeneity before correction: "
          f"MAD CV={het['mad_cv']}, median CV={het['median_cv']}, "
          f"source-fraction CV={het['source_fraction_cv']}")

    results = {}
    for name, fn in NORMS.items():
        same = [persistence_diagram(fn(img), 0) for _, img in at_field]
        other = [persistence_diagram(fn(img), 0) for _, img in off_field]
        within = [bottleneck(same[i], same[j])
                  for i in range(len(same)) for j in range(i + 1, len(same))]
        between = [bottleneck(same[i], other[j])
                   for i in range(len(same)) for j in range(len(other))]
        w, b = float(np.median(within)), float(np.median(between))
        # Separation expressed relative to the within-field scale, so it is comparable
        # across normalisations that live on completely different filtration axes.
        sep = (b - w) / max(w, 1e-12)
        results[name] = {
            "n_within": len(within), "n_between": len(between),
            "median_within": round(w, 6), "median_between": round(b, 6),
            "relative_separation": round(sep, 4),
            "mean_diagram_size": round(float(np.mean(
                [diagram_summary(d)["n"] for d in same])), 1),
        }
        print(f"  {name:8s} within={w:12.4f} between={b:12.4f} "
              f"relative separation={sep:+.4f}")

    best = max(results, key=lambda k: results[k]["relative_separation"])
    baseline_sep = results["none"]["relative_separation"]
    best_sep = results[best]["relative_separation"]

    # The gate. A separation of a couple of percent is what the unnormalised run already
    # produced and was judged meaningless; the threshold is set well above it and stated
    # here rather than chosen after seeing the numbers.
    THRESHOLD = 0.15
    verdict = {
        "best_normalisation": best,
        "relative_separation_none": baseline_sep,
        "relative_separation_best": best_sep,
        "improvement": round(best_sep - baseline_sep, 4),
        "threshold": THRESHOLD,
        "gate_passes": bool(best_sep >= THRESHOLD),
        "interpretation": (
            "Sublevel-set persistence depends only on the order pixels enter the "
            "filtration, so a rank transform is the canonical representative of the "
            "monotone class the method is already shape-invariant to. If even that fails "
            "to separate same-field from different-field diagrams, the topology of these "
            "plates is dominated by grain and defects whose ordering also varies plate to "
            "plate, and no monotone correction can repair it."),
    }
    print(f"\nbest: {best} at {best_sep:+.4f} relative separation "
          f"(unnormalised {baseline_sep:+.4f}, threshold {THRESHOLD})")
    print(f"GATE {'PASSES' if verdict['gate_passes'] else 'FAILS'}")
    if not verdict["gate_passes"]:
        print("  The temporal-topology premise does not survive. T4 closes with this "
              "recorded as the finding.")

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "normalisation_gate.json").write_text(json.dumps({
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "seed": SEED, "patch_pixels": PATCH, "offset_deg": args.offset,
        "plate_heterogeneity": het,
        "plates": meta_rows,
        "results": results,
        "verdict": verdict,
    }, indent=2))
    plates.to_csv(out / "normalisation_gate_plates.csv", index=False)
    return 0 if verdict["gate_passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
