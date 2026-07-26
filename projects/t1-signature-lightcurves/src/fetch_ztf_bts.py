"""Assemble a labelled ZTF light-curve sample: BTS spectroscopic classes + ALeRCE photometry.

Labels come from the ZTF Bright Transient Survey sample explorer, which publishes
spectroscopic classifications as a flat CSV. Per-epoch photometry comes from the ALeRCE
broker, which serves individual detections without authentication.

The two sources are deliberately kept apart. BTS classifications are spectroscopic and are
the ground truth; ALeRCE also publishes its own light-curve-classifier labels, and those
are never used here, because training on one classifier's output and reporting agreement
with it measures nothing.

Nothing is interpolated, binned or resampled. The output is the raw
(time, magnitude, uncertainty, band) stream, which is what the signature construction
consumes.

Run:  python3 src/fetch_ztf_bts.py [--per-class 500] [--workers 12]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

BTS_URL = "https://sites.astro.caltech.edu/ztf/bts/explorer.php?f=s&format=csv"
ALERCE = "https://api.alerce.online/ztf/v1/objects/{oid}/detections"
FID_TO_BAND = {1: "g", 2: "r", 3: "i"}

# Classes retained, with the rarer supernova subtypes kept separate rather than merged
# into their parent class. Merging would make the problem easier and less interesting:
# distinguishing SN Ia from SN Ia-91T is exactly the kind of shape-and-ordering
# distinction the representation is being tested on.
CLASSES = ["SN Ia", "SN II", "CV", "AGN", "SN IIn", "SN Ic", "SN Ib", "SLSN-I"]

MIN_DETECTIONS = 10     # below this a light curve carries too little shape to classify
MIN_PER_BAND = 3        # require both bands to be genuinely sampled
ROOT = Path(__file__).resolve().parent.parent


def fetch_bts(cache: Path) -> pd.DataFrame:
    """Download the BTS catalogue, caching the raw CSV for reproducibility."""
    if cache.exists():
        print(f"BTS catalogue from cache: {cache}")
        return pd.read_csv(cache)
    print("downloading BTS catalogue ...")
    resp = requests.get(BTS_URL, timeout=120)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"  {len(df)} rows -> {cache}")
    return df


def select_sample(bts: pd.DataFrame, per_class: int, seed: int = 20260726) -> pd.DataFrame:
    """Take a class-balanced subset of the spectroscopically classified objects."""
    bts = bts.copy()
    bts["type"] = bts["type"].astype(str).str.strip()
    classified = bts[bts["type"] != "-"]
    print(f"spectroscopically classified: {len(classified)} of {len(bts)}")

    keep = classified[classified["type"].isin(CLASSES)]
    sample = (
        keep.groupby("type", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_class), random_state=seed),
               include_groups=True)
        .reset_index(drop=True)
    )
    # Shuffle before any network work. A first run submitted objects grouped by class and
    # the broker's failure rate rose over the course of the run, so the classes submitted
    # last lost a far larger fraction of their objects than those submitted first -- SN Ib
    # was wiped out entirely. That is a selection effect on the label, not random
    # attrition, and it would have corrupted every downstream number. Shuffling makes any
    # remaining attrition independent of class.
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print("class balance of the selected sample:")
    for cls, n in sample["type"].value_counts().items():
        print(f"  {cls:10s} {n:5d}")
    return sample


def fetch_one(oid: str, cache_dir: Path, retries: int = 5) -> list[dict] | None:
    """Fetch one object's detections, with an on-disk cache and exponential backoff."""
    cached = cache_dir / f"{oid}.json"
    if cached.exists():
        try:
            return json.loads(cached.read_text())
        except json.JSONDecodeError:
            cached.unlink(missing_ok=True)

    for attempt in range(retries):
        try:
            resp = requests.get(ALERCE.format(oid=oid), timeout=60)
            if resp.status_code == 200:
                det = resp.json()
                cached.write_text(json.dumps(det))
                return det
            if resp.status_code == 404:
                cached.write_text("[]")  # genuinely absent, do not retry on later runs
                return None
            if resp.status_code == 429:  # rate limited: back off hard
                time.sleep(5.0 * (attempt + 1))
                continue
        except requests.RequestException:
            pass
        time.sleep(2.0 * (2 ** attempt))  # 2, 4, 8, 16, 32 s
    return None


def fetch_photometry(sample: pd.DataFrame, workers: int, cache_dir: Path
                     ) -> tuple[pd.DataFrame, dict]:
    """Pull per-epoch photometry for every object, in parallel, resuming from cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    labels = dict(zip(sample["ZTFID"], sample["type"]))
    rows: list[dict] = []
    failed: list[str] = []
    t0 = time.time()
    n_cached = sum(1 for oid in sample["ZTFID"] if (cache_dir / f"{oid}.json").exists())
    if n_cached:
        print(f"  {n_cached}/{len(sample)} objects already cached")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, oid, cache_dir): oid for oid in sample["ZTFID"]}
        for i, fut in enumerate(as_completed(futures), 1):
            oid = futures[fut]
            det = fut.result()
            if not det:
                failed.append(oid)
            else:
                for d in det:
                    band = FID_TO_BAND.get(d.get("fid"))
                    mag, err, mjd = d.get("magpsf"), d.get("sigmapsf"), d.get("mjd")
                    if band is None or mag is None or err is None or mjd is None:
                        continue
                    rows.append({"oid": oid, "label": labels[oid], "mjd": mjd,
                                 "band": band, "mag": mag, "magerr": err})
            if i % 250 == 0:
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(futures)} objects ({rate:.1f}/s, {len(failed)} failed)")

    df = pd.DataFrame(rows)
    stats = {"requested": len(sample), "failed": len(failed),
             "elapsed_s": round(time.time() - t0, 1)}
    print(f"fetched {len(df)} detections for {df.oid.nunique()} objects "
          f"in {stats['elapsed_s']} s ({len(failed)} failures)")

    # Attrition must not correlate with class. Report the per-class retrieval rate so the
    # check is visible rather than assumed.
    got = set(df.oid.unique()) if not df.empty else set()
    rates = (sample.assign(ok=sample.ZTFID.isin(got))
             .groupby("type").ok.agg(["sum", "count"]))
    rates["rate"] = (rates["sum"] / rates["count"]).round(3)
    print("retrieval rate by class:")
    print(rates.to_string())
    stats["retrieval_rate_by_class"] = rates["rate"].to_dict()
    spread = float(rates["rate"].max() - rates["rate"].min())
    stats["retrieval_rate_spread"] = round(spread, 3)
    if spread > 0.15:
        print(f"  WARNING: retrieval rate varies by {spread:.2f} across classes; "
              f"attrition may be class-dependent")
    return df, stats


def apply_quality_cuts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop light curves too sparse to carry shape information. Cuts are reported."""
    before_obj = df.oid.nunique()

    counts = df.groupby("oid").size()
    enough_total = counts[counts >= MIN_DETECTIONS].index
    df = df[df.oid.isin(enough_total)]

    per_band = df.groupby(["oid", "band"]).size().unstack(fill_value=0)
    both_bands = per_band[(per_band.get("g", 0) >= MIN_PER_BAND)
                          & (per_band.get("r", 0) >= MIN_PER_BAND)].index
    df = df[df.oid.isin(both_bands)]

    df = df.sort_values(["oid", "mjd"]).reset_index(drop=True)
    cuts = {
        "objects_before": int(before_obj),
        "objects_after": int(df.oid.nunique()),
        "dropped_too_few_detections": int(before_obj - len(enough_total)),
        "dropped_single_band": int(len(enough_total) - len(both_bands)),
        "detections_after": int(len(df)),
    }
    print(f"quality cuts: {cuts['objects_before']} -> {cuts['objects_after']} objects "
          f"({cuts['dropped_too_few_detections']} too sparse, "
          f"{cuts['dropped_single_band']} effectively single-band)")
    return df, cuts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=500,
                    help="maximum objects per class before quality cuts")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    data = ROOT / "data"
    outputs = ROOT / "outputs"
    data.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)

    bts = fetch_bts(data / "bts_explorer.csv")
    sample = select_sample(bts, args.per_class)

    photo, fetch_stats = fetch_photometry(sample, args.workers, data / "alerce_cache")
    if photo.empty:
        print("no photometry retrieved", file=sys.stderr)
        return 1

    photo, cuts = apply_quality_cuts(photo)
    photo.to_parquet(data / "ztf_bts_lightcurves.parquet", index=False)

    # The manifest is small and is committed, so the exact sample is reproducible even
    # if the upstream catalogue changes.
    manifest = (photo.groupby("oid")
                .agg(label=("label", "first"), n_det=("mjd", "size"),
                     t_span=("mjd", lambda s: round(s.max() - s.min(), 3)))
                .reset_index())
    manifest.to_csv(outputs / "sample_manifest.csv", index=False)

    summary = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "classes": CLASSES,
        "per_class_cap": args.per_class,
        "min_detections": MIN_DETECTIONS,
        "min_per_band": MIN_PER_BAND,
        "fetch": fetch_stats,
        "quality_cuts": cuts,
        "class_counts": manifest.label.value_counts().to_dict(),
        "median_detections": int(manifest.n_det.median()),
        "median_span_days": float(manifest.t_span.median()),
    }
    (outputs / "fetch_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["class_counts"], indent=2))
    print(f"median detections per object: {summary['median_detections']}, "
          f"median span: {summary['median_span_days']:.1f} d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
