"""Second gate: does the topology see CHANGE, or only place?

The first gate established that z-score-normalised persistence diagrams separate one field
from another across plates, at a relative separation of 0.628. That is necessary and not
sufficient. Telling two patches of sky apart is not the same as telling one patch of sky
apart from ITSELF at a different epoch, and the second is the entire point of working a
century-long archive.

THE TEST. Take a field containing a source known to have changed dramatically within the
plate era, and a control field with no such source, observed on the same plates. Compare:

    target  epoch-to-epoch distance within the CHANGING field
    control epoch-to-epoch distance within the STATIC field

If the topology sees change, the target field should be less self-consistent across epochs
than the control field is. The statistic is the ratio of the two, and the field-to-field
separation of 0.628 is the scale it has to be read against: a change signal smaller than the
difference between two arbitrary patches of sky is not usable for detecting change.

TARGETS, chosen because the plate era covers their variation and their behaviour is not in
dispute:

  GK Per (Nova Persei 1901)  Erupted 1901 February to magnitude 0.2 from a quiescent 13,
                             one of the brightest novae on record, and DASCH coverage begins
                             in the 1880s. Before-and-after plates exist by construction.

  omicron Ceti (Mira)        Pulsates over roughly 8 magnitudes on a 332-day period. Any
                             pair of plates separated by a few months brackets a large
                             change, so this target tests the method on recurring variation
                             rather than a single event.

WHAT WOULD FAIL IT. If target and control fields are equally self-consistent across epochs,
the diagrams are tracking plate properties and field content but not temporal change, and
the temporal-topology premise fails on its second and stricter test. That outcome closes the
track, and it is a real possibility: the first gate's separation could be driven entirely by
which stars happen to sit in a field, a property that does not change with epoch.

Run:  python3 src/temporal_gate.py [--target gkper] [--n-plates 16]
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
from falsify_grain import PATCH, bottleneck, persistence_diagram  # noqa: E402
from normalisation_gate import norm_zscore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260803

# Field-to-field separation measured by the first gate under the same normalisation. Any
# change signal has to be read against this scale.
FIELD_BASELINE = 0.6276

TARGETS = {
    "gkper": {
        "name": "GK Persei (Nova Persei 1901)",
        "ra": 52.80, "dec": 43.90,
        "why": ("erupted February 1901 from quiescent magnitude 13 to 0.2; DASCH coverage "
                "begins in the 1880s, so before-and-after plates exist by construction"),
    },
    "mira": {
        "name": "omicron Ceti (Mira)",
        "ra": 34.8366, "dec": -2.9776,
        "why": ("pulsates over roughly 8 magnitudes on a 332-day period, so plates months "
                "apart bracket a large change; tests recurring variation, not one event"),
    },
}
CONTROL_OFFSET = 0.8  # degrees; same plates, no known variable of comparable amplitude


def collect(ra: float, dec: float, n_plates: int, offset: float
            ) -> tuple[list[tuple[str, str, np.ndarray]], list[tuple[str, str, np.ndarray]]]:
    """Cutouts at the target position and at a control offset, from the same plates."""
    ids = search_plates(ra, dec)
    print(f"  {len(ids)} plates cover the position")
    target, control = [], []
    for pid in ids:
        if len(target) >= n_plates:
            break
        meta = plate_metadata(pid)
        if meta is None:
            continue
        exps = exposures(meta)
        if not exps or exps[0].get("ctr_ra") is None:
            continue
        epoch = plate_epoch(meta) or "?"
        t = cutout(pid, ra, dec, 0)
        if t is None:
            continue
        c = cutout(pid, ra + offset, dec, 0)
        if c is None:
            continue
        target.append((pid, epoch, t[0][:PATCH, :PATCH]))
        control.append((pid, epoch, c[0][:PATCH, :PATCH]))
        print(f"    {pid} {epoch}")
    return target, control


def self_distances(frames: list[tuple[str, str, np.ndarray]]) -> tuple[list[float], list]:
    """Pairwise bottleneck distances between epochs of the same field, z-score normalised."""
    diags = [persistence_diagram(norm_zscore(img), 0) for _, _, img in frames]
    pairs = []
    for i in range(len(diags)):
        for j in range(i + 1, len(diags)):
            d = bottleneck(diags[i], diags[j])
            pairs.append({"a": frames[i][0], "b": frames[j][0],
                          "epoch_a": frames[i][1], "epoch_b": frames[j][1],
                          "distance": round(d, 6)})
    return [p["distance"] for p in pairs], pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gkper", choices=sorted(TARGETS))
    ap.add_argument("--n-plates", type=int, default=16)
    ap.add_argument("--offset", type=float, default=CONTROL_OFFSET)
    args = ap.parse_args()

    spec = TARGETS[args.target]
    print(f"target: {spec['name']}")
    print(f"  {spec['why']}")
    print(f"  control field offset {args.offset} deg, same plates\n")

    target, control = collect(spec["ra"], spec["dec"], args.n_plates, args.offset)
    if len(target) < 4:
        print("fewer than four usable plates at this position; cannot evaluate",
              file=sys.stderr)
        return 1

    t_dists, t_pairs = self_distances(target)
    c_dists, c_pairs = self_distances(control)
    t_med, c_med = float(np.median(t_dists)), float(np.median(c_dists))

    # How much less self-consistent is the changing field than the static one, relative to
    # the static field's own scale? Positive means change is visible.
    change_signal = (t_med - c_med) / max(c_med, 1e-12)

    print(f"\nepoch-to-epoch bottleneck distance, {len(t_dists)} pairs each")
    print(f"  target field  (contains {spec['name']}): median {t_med:.4f}")
    print(f"  control field (offset {args.offset} deg):  median {c_med:.4f}")
    print(f"  change signal: {change_signal:+.4f}")
    print(f"  field-to-field baseline from gate 1: {FIELD_BASELINE:+.4f}")

    # The bar: a change signal has to be comparable to the scale on which the method already
    # distinguishes two different patches of sky. Fixed here, before interpreting the run.
    THRESHOLD = 0.5 * FIELD_BASELINE
    verdict = {
        "target": args.target, "target_name": spec["name"],
        "n_plates": len(target), "n_pairs": len(t_dists),
        "target_median": round(t_med, 6), "control_median": round(c_med, 6),
        "change_signal": round(change_signal, 4),
        "field_baseline": FIELD_BASELINE,
        "threshold": round(THRESHOLD, 4),
        "gate_passes": bool(change_signal >= THRESHOLD),
        "interpretation": (
            "The changing field must be measurably less self-consistent across epochs than "
            "a static field on the same plates. If it is not, the diagrams track plate "
            "properties and field content but not time, and the temporal premise fails."),
    }
    print(f"\nGATE 2 {'PASSES' if verdict['gate_passes'] else 'FAILS'} "
          f"(threshold {THRESHOLD:+.4f})")
    if not verdict["gate_passes"]:
        print("  The method distinguishes places, not times. T4's temporal premise does "
              "not survive its second test.")

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / f"temporal_gate_{args.target}.json").write_text(json.dumps({
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "seed": SEED, "patch_pixels": PATCH, "normalisation": "zscore",
        "target_spec": spec, "control_offset_deg": args.offset,
        "plates": [{"plate_id": p, "epoch": e} for p, e, _ in target],
        "target_pairs": t_pairs, "control_pairs": c_pairs,
        "verdict": verdict,
    }, indent=2))
    return 0 if verdict["gate_passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
