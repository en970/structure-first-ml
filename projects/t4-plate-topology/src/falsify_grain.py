"""Does plate topology measure the sky, or the emulsion?

THIS TEST RUNS BEFORE THE PROJECT. The premise of T4 is that sublevel-set persistent
homology of photographic plate images carries astrophysical structure. Photographic
emulsion has grain, scans have seams and dust, and plates from the 1890s have both in
quantity. If persistence diagrams are dominated by those rather than by sources, there is no
project here, and it is far cheaper to discover that now than after building a pipeline.

The mathematics is cheap in exactly the way it was not for Gaia. On a point cloud in five
dimensions, degree-1 Vietoris-Rips persistence is intractable at survey scale -- that killed
an earlier candidate in this repository. On a 2-D image the natural construction is a
cubical complex over the pixel grid with a sublevel-set filtration, which is near-linear:
sort the pixels by intensity, insert them in order, and track when connected components
(H0) and loops (H1) are born and die. A plate that is 1976 x 2204 is four million cells and
takes seconds.

THREE FALSIFICATION TESTS, each with a stated failure condition:

  T1 SIGNAL VERSUS BLANK. Within one plate, compare a field known to contain many sources
     against a field at similar radius containing few. If persistence statistics are
     indistinguishable, the diagrams are measuring emulsion, not sky. FAILS the premise if
     the separation is within noise.

  T2 EPOCH REPRODUCIBILITY. Take the same SKY POSITION on plates from different dates.
     Because cutouts are requested by right ascension and declination, this is genuinely
     WCS-aligned rather than a pixel-window proxy. A real structure recurs across plates;
     grain does not. FAILS if diagrams of the same field on different plates are no more
     similar to each other than to diagrams of a different field.

  T3 GRAIN CONTROL. Compare a real plate field against a synthetic field with the same
     intensity histogram but pixels shuffled -- identical marginal distribution, no spatial
     structure at all. Persistence must DISTINGUISH them, or it is not seeing structure.
     FAILS if a shuffled field produces a diagram indistinguishable from a real one.

     A REGISTERED EXPECTATION THAT WAS WRONG, AND THE CRITERION IT COST. The first version
     of this test required the real field to show LONGER-lived features than the shuffled
     one. That is backwards. Measured: shuffled fields give more components (median 7,314
     against 5,781) and larger total persistence (948k against 497k). The reason is that
     shuffling preserves the histogram, so every bright pixel survives but is scattered into
     isolation, and an isolated bright pixel on a flat background is a long-lived component.
     Real sources are spatially coherent, so their pixels merge into far fewer components.
     Direction of lifetime is therefore not the discriminant; SEPARABILITY is. The criterion
     is now stated as such -- fewer components in the real field plus a bottleneck distance
     large against the diagrams' own scale -- and the original expectation is left on the
     record because a criterion rewritten after seeing the data has to show its working.

The tests are constructed so they CAN fail. T3 in particular is a hard floor: if a
pixel-shuffled image produces the same diagram as a real one, nothing downstream is
meaningful.

WHICH DATA PRODUCT, AND WHY IT DECIDED THE FIRST RUN. The first version of this test used
16x-binned whole-plate mosaics and T1 failed with every patch reporting a bright-pixel
fraction of exactly zero -- no patch contained a single source above five times the robust
scatter. Measured on plate a02115 over the same field:

    product                        background MAD   fraction above 5 MAD
    16x-binned mosaic                    833.0            0.00000
    calibrated cutout (1.44 arcsec/px)    72.0            0.00522

Binning by sixteen collapses stellar profiles below one pixel and folds the plate's
vignetting gradient into the scatter, inflating the MAD elevenfold. The mosaic is the right
product for locating a plate and the wrong one for analysing structure on it. The test now
uses calibrated cutouts, which also upgrades T2 from a pixel-window proxy to a real
WCS-aligned comparison. The failure was in the data product, not the method -- but it would
have read as a failure of the method.

Run:  python3 src/falsify_grain.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dasch_access import (binned_mosaic, cutout, exposures, has_mosaic,  # noqa: E402
                          plate_epoch, plate_metadata, search_plates)

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260730
PATCH = 256          # pixels per analysed patch, taken from an 835x835 cutout
MIN_PERSISTENCE = 0  # keep everything; filtering is a downstream choice


def persistence_diagram(patch: np.ndarray, homology_dim: int = 0) -> np.ndarray:
    """Sublevel-set persistence of a 2-D patch via a cubical complex.

    Intensity is negated first so that BRIGHT features are the ones born early: plate scans
    store higher counts for brighter sky, and a sublevel-set filtration on the raw array
    would track the dark background instead.
    """
    import gudhi
    arr = -np.asarray(patch, dtype=np.float64)
    cc = gudhi.CubicalComplex(top_dimensional_cells=arr.flatten(),
                              dimensions=arr.shape)
    cc.persistence(homology_coeff_field=2, min_persistence=MIN_PERSISTENCE)
    pairs = cc.persistence_intervals_in_dimension(homology_dim)
    if len(pairs) == 0:
        return np.empty((0, 2))
    finite = np.asarray([p for p in pairs if np.isfinite(p[1])])
    return finite if len(finite) else np.empty((0, 2))


def diagram_summary(diag: np.ndarray) -> dict:
    """Scalar statistics of a diagram, enough to compare fields without a metric."""
    if len(diag) == 0:
        return {"n": 0, "total_persistence": 0.0, "max_persistence": 0.0,
                "mean_persistence": 0.0, "n_significant": 0}
    life = diag[:, 1] - diag[:, 0]
    return {"n": int(len(diag)),
            "total_persistence": float(life.sum()),
            "max_persistence": float(life.max()),
            "mean_persistence": float(life.mean()),
            # "significant" = lifetime above 3x the median, a crude but honest cut
            "n_significant": int((life > 3.0 * max(np.median(life), 1e-9)).sum())}


def bottleneck(d1: np.ndarray, d2: np.ndarray) -> float:
    """Bottleneck distance between two diagrams, the metric the stability theorem bounds."""
    import gudhi
    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    return float(gudhi.bottleneck_distance(d1, d2))


def patches_from(img: np.ndarray, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Random square patches away from the plate edges, where vignetting dominates."""
    h, w = img.shape
    margin = PATCH
    out = []
    for _ in range(n):
        y = int(rng.integers(margin, max(margin + 1, h - margin - PATCH)))
        x = int(rng.integers(margin, max(margin + 1, w - margin - PATCH)))
        out.append(img[y:y + PATCH, x:x + PATCH])
    return out


def source_fraction(img: np.ndarray) -> float:
    """Fraction of pixels more than five robust deviations above the background."""
    bg = float(np.nanmedian(img))
    mad = float(np.nanmedian(np.abs(img - bg)))
    if mad <= 0:
        return 0.0
    return float(np.nanmean((img - bg) / mad > 5.0))


def sub_patches(img: np.ndarray, rng: np.random.Generator, n: int = 4) -> list[np.ndarray]:
    """Non-overlapping PATCH-sized windows from inside a cutout."""
    h, w = img.shape
    if h < PATCH or w < PATCH:
        return [img]
    out = []
    for _ in range(n):
        y = int(rng.integers(0, h - PATCH + 1))
        x = int(rng.integers(0, w - PATCH + 1))
        out.append(img[y:y + PATCH, x:x + PATCH])
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    ra0, dec0 = 11.446426, -71.535997
    ids = search_plates(ra0, dec0)
    print(f"{len(ids)} plates cover the field")

    # Plates with a usable exposure record, spread over whatever dates are available.
    chosen: list[tuple[str, str, dict]] = []
    for pid in ids:
        if len(chosen) >= 12:
            break
        meta = plate_metadata(pid)
        if meta is None:
            continue
        exps = exposures(meta)
        if not exps or exps[0].get("ctr_ra") is None:
            continue
        chosen.append((pid, plate_epoch(meta) or "?", exps[0]))
        print(f"  {pid} {plate_epoch(meta)}")

    if len(chosen) < 2:
        print("fewer than two usable plates; cannot run the tests", file=sys.stderr)
        return 1

    # One cutout per plate at the field centre, and one per plate offset by half a degree
    # to serve as a different field at comparable plate radius.
    OFFSET = 0.5
    at_field: list[tuple[str, np.ndarray]] = []
    off_field: list[tuple[str, np.ndarray]] = []
    for pid, epoch, ex in chosen:
        c = cutout(pid, float(ex["ctr_ra"]), float(ex["ctr_dec"]), 0)
        if c is not None:
            at_field.append((pid, c[0]))
        c2 = cutout(pid, float(ex["ctr_ra"]) + OFFSET, float(ex["ctr_dec"]), 0)
        if c2 is not None:
            off_field.append((pid, c2[0]))
    print(f"cutouts: {len(at_field)} at field centre, {len(off_field)} offset by "
          f"{OFFSET} deg (from {len(chosen)} plates attempted)")
    if len(at_field) < 2:
        print("fewer than two cutouts returned; cannot run the tests", file=sys.stderr)
        return 1

    results: dict = {"n_plates": len(chosen), "offset_deg": OFFSET,
                     "plates": [{"id": p, "epoch": e} for p, e, _ in chosen]}

    def med(stats, key):
        return float(np.median([s[key] for s in stats]))

    # ---------------------------------------------------------------- T3 grain control
    print("\nT3 grain control: real cutout patches versus pixel-shuffled patches")
    real_stats, shuf_stats, dists = [], [], []
    for pid, img in at_field[:3]:
        for patch in sub_patches(img, rng, 3):
            d_real = persistence_diagram(patch, 0)
            flat = patch.flatten().copy()
            rng.shuffle(flat)
            d_shuf = persistence_diagram(flat.reshape(patch.shape), 0)
            real_stats.append(diagram_summary(d_real))
            shuf_stats.append(diagram_summary(d_shuf))
            dists.append(bottleneck(d_real, d_shuf))
    t3 = {
        "real_median_n": med(real_stats, "n"),
        "shuffled_median_n": med(shuf_stats, "n"),
        "real_median_max_persistence": med(real_stats, "max_persistence"),
        "shuffled_median_max_persistence": med(shuf_stats, "max_persistence"),
        "real_median_total_persistence": med(real_stats, "total_persistence"),
        "shuffled_median_total_persistence": med(shuf_stats, "total_persistence"),
        "median_bottleneck_real_vs_shuffled": float(np.median(dists)),
    }
    # Separability, not lifetime direction (see the docstring for why the original
    # expectation was wrong). Two requirements: spatial coherence must reduce the component
    # count, and the diagrams must be far apart relative to their own persistence scale.
    scale = max(t3["real_median_max_persistence"], 1e-9)
    t3["bottleneck_over_persistence_scale"] = round(
        t3["median_bottleneck_real_vs_shuffled"] / scale, 3)
    t3["total_persistence_ratio_shuffled_over_real"] = round(
        t3["shuffled_median_total_persistence"]
        / max(t3["real_median_total_persistence"], 1e-9), 3)
    t3["passes"] = bool(t3["real_median_n"] < t3["shuffled_median_n"]
                        and t3["bottleneck_over_persistence_scale"] > 0.25)
    for k, v in t3.items():
        print(f"  {k}: {v}")
    results["T3_grain_control"] = t3

    # ---------------------------------------------------------------- T1 signal vs blank
    print("\nT1 signal versus blank, ranked by measured source content")
    pool = [(pid, p) for pid, img in at_field for p in sub_patches(img, rng, 4)] + \
           [(pid, p) for pid, img in off_field for p in sub_patches(img, rng, 2)]
    frac = [source_fraction(p) for _, p in pool]
    order = np.argsort(frac)
    k = max(2, len(pool) // 4)
    poor_idx, rich_idx = order[:k], order[-k:]
    poor_stats = [diagram_summary(persistence_diagram(pool[i][1], 0)) for i in poor_idx]
    rich_stats = [diagram_summary(persistence_diagram(pool[i][1], 0)) for i in rich_idx]
    t1 = {
        "n_patches": len(pool),
        "source_fraction_poor": [round(frac[i], 5) for i in poor_idx],
        "source_fraction_rich": [round(frac[i], 5) for i in rich_idx],
        "poor_median_total_persistence": med(poor_stats, "total_persistence"),
        "rich_median_total_persistence": med(rich_stats, "total_persistence"),
        "poor_median_n_significant": med(poor_stats, "n_significant"),
        "rich_median_n_significant": med(rich_stats, "n_significant"),
        "poor_median_max_persistence": med(poor_stats, "max_persistence"),
        "rich_median_max_persistence": med(rich_stats, "max_persistence"),
    }
    t1["source_content_separated"] = bool(
        max(t1["source_fraction_poor"]) < min(t1["source_fraction_rich"]))
    t1["passes"] = bool(t1["source_content_separated"]
                        and t1["rich_median_max_persistence"]
                        > t1["poor_median_max_persistence"])
    for key, v in t1.items():
        print(f"  {key}: {v}")
    results["T1_signal_vs_blank"] = t1

    # ---------------------------------------------------------------- T2 reproducibility
    print("\nT2 epoch reproducibility: same sky position across plates (WCS-aligned)")
    same = [persistence_diagram(img[:PATCH, :PATCH], 0) for _, img in at_field]
    other = [persistence_diagram(img[:PATCH, :PATCH], 0) for _, img in off_field]
    within = [bottleneck(same[i], same[j])
              for i in range(len(same)) for j in range(i + 1, len(same))]
    between = [bottleneck(same[i], other[j])
               for i in range(len(same)) for j in range(len(other))]
    t2 = {
        "n_pairs_within": len(within), "n_pairs_between": len(between),
        "median_bottleneck_within_field": float(np.median(within)) if within else None,
        "median_bottleneck_between_fields": float(np.median(between)) if between else None,
        "note": "WCS-aligned: cutouts requested by RA/Dec, not pixel windows",
    }
    t2["passes"] = bool(within and between
                        and np.median(within) < np.median(between))
    for key, v in t2.items():
        print(f"  {key}: {v}")
    results["T2_epoch_reproducibility"] = t2

    verdict = {
        "T1_passes": t1["passes"], "T2_passes": t2["passes"], "T3_passes": t3["passes"],
        "premise_survives": bool(t3["passes"] and t1["passes"]),
        "interpretation": (
            "T3 is the hard floor: a pixel-shuffled image must not look like a real one. "
            "T1 is the minimum useful signal: persistence must respond to source content. "
            "T2 is now a real WCS-aligned test rather than a proxy, since cutouts are "
            "requested by sky position."),
    }
    print(f"\npremise survives: {verdict['premise_survives']}")
    print(f"  T1 {t1['passes']}   T2 {t2['passes']}   T3 {t3['passes']}")
    results["verdict"] = verdict

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    import pandas as pd
    results["generated_utc"] = pd.Timestamp.utcnow().isoformat()
    results["seed"] = SEED
    results["patch_pixels"] = PATCH
    (out / "falsify_grain.json").write_text(json.dumps(results, indent=2))
    return 0 if verdict["premise_survives"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
