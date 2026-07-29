"""Assemble the T1 counter-test sample: Gaia DR3 periodic variables with epoch photometry.

WHY THIS SAMPLE EXISTS
----------------------
T1's primary sample is ZTF transients, which are distinguished by the shape and ordering of
the light curve. That is what the signature transform encodes, and on it signatures came
within 0.014 balanced accuracy of a hand-crafted baseline (0.547 against 0.562) while
carrying complementary information.

Gaia DR3 periodic variables are the opposite case, and are here because the method should
handle them badly. An eclipsing binary, an RR Lyrae, a Cepheid and a spotted rotator are
separated chiefly by *period*, and period is a statement about the clock. The signature is
invariant to increasing reparameterisation of the clock, so a plain signature should discard
exactly the discriminant. Two predictions follow, both recorded here before the experiment
is run so that either can fail on the record:

  P1. Signatures should lose to the hand-crafted baseline by a markedly LARGER margin on
      this sample than the 0.014 measured on ZTF.
  P2. Adding an explicit time channel should recover much MORE here than the 0.069 it
      recovered on ZTF, because here the clock is the signal rather than the weather.

If signatures instead win here too, for the reasons claimed on ZTF, then the account of the
mechanism given in T1 is wrong and the track's headline has to change. A method that wins
everywhere for the same stated reason has not been understood.

WHAT WAS MEASURED, NOT ASSUMED
------------------------------
Every number in this section came from a live probe of the ESA Gaia archive from this
machine, on 2026-07-30, and each is a constraint on the design of this script.

  * `gaiadr3.vari_classifier_result` holds 9,976,881 rows, one classifier
    (`nTransits:5+`), 24 distinct `best_class_name` values.
  * ALL 9,976,881 of those sources carry `has_epoch_photometry = 'true'` in
    `gaiadr3.gaia_source`, so epoch photometry is never the reason a source is missing.
    Attrition, if it appears, is a transport failure and must be treated as one.
  * DATALINK `DATA_STRUCTURE` accepts only `RAW` and `INDIVIDUAL` for EPOCH_PHOTOMETRY.
    `COMBINED`, which the project's data-source note implied, is rejected with
    "Invalid Data Structure parameter: 'COMBINED', valid ones are: RAW, INDIVIDUAL".
  * The 5,000-identifier cap per DATALINK call is exact, not approximate: 5,000 ids
    returned HTTP 200 and a ZIP; 5,001 returned HTTP 500. Verified at the HTTP layer.
  * `INDIVIDUAL` + `FORMAT=csv` returns 25 columns, one row per transit, with a SEPARATE
    timestamp per band (`g_transit_time`, `bp_obs_time`, `rp_obs_time`). The wide-to-long
    reshape below is therefore mandatory, and the per-band timestamps are kept distinct
    because the three bands genuinely are sampled at different instants within a transit.
  * Measured 6,140 bytes per source compressed; 33 to 51 transit rows per source.
  * The DATALINK stream is throttled to roughly 1.7 kB/s per connection: 25 ids took
    127.7 s serially (0.196 sources/s). Eight concurrent connections gave 0.867 sources/s
    (4.43x), sixteen gave 1.113 sources/s. Throughput plateaus near 1.1 sources/s, so the
    fetch is parallel by necessity and a 3,000-object sample costs roughly one to 1.5 hours.
  * Class availability at `best_class_score >= 0.8`, which is what makes a balanced sample
    of the six retained classes possible at all: LPV 710,604; ECL 535,660;
    DSCT|GDOR|SXPHE 90,335; RR 70,469; RS 63,477; CEP 2,569. CEP is the binding constraint,
    so `--per-class` above about 2,500 cannot stay balanced. At `>= 0.9` CEP falls to 854
    and BCEP and SPB collapse to double digits, which is why 0.8 is the default floor.

Sampling is by `ORDER BY gaia_source.random_index`, not by `source_id`. `source_id` encodes
HEALPix position, so ordering by it returns a single patch of sky: an early probe without
`ORDER BY` returned 12 identifiers all sharing a three-digit prefix. With `random_index`,
2,870 of 3,000 ECL identifiers had distinct level-6 HEALPix prefixes. The bias was real and
is removed deliberately.

Nothing is interpolated, binned or resampled. Output is the raw
(time, magnitude, uncertainty, band) stream the signature construction consumes.

PILOT RESULT, 2026-07-30
------------------------
`--per-class 50 --workers 12 --batch 25 --tag pilot`, i.e. 300 objects across the six
classes, all of it from live queries:

  * 300 of 300 sources retrieved, zero failures, per-class retrieval rate 1.000 for all six
    classes, spread 0.000. The class-attrition check passed with nothing to report.
  * 477.6 s wall, 0.63 sources/s at 12 workers, so 3,000 objects is roughly 80 minutes.
  * 14,553 transit rows -> 43,659 band-slots -> 40,154 observations. Of the slots, 2,290
    (5.2 per cent) had no photometry in that band for that transit and 1,215 (2.8 per cent)
    carried a variability reject flag.
  * Median 127 observations per object (p10-p90 75 to 190) over a median 961.3 d baseline.
    Quality cuts removed nothing: 300 -> 300.
  * MJD range 56863.47 to 57901.33, inside the DR3 window, so the TCB offset is right.
  * Within a single transit the BP timestamp trails G by a median 27.2 s and RP by 34.8 s.
    The three bands are genuinely not simultaneous, which is why the reshape keeps three
    timestamps rather than one.
  * The merged inter-observation gap distribution spans five orders of magnitude:
    p50 0.0003 d (the within-transit band offsets), p75 0.074 d, p90 30.0 d, p99 129.0 d.
    Wider than the four orders in the ZTF sample.
  * Catalogued median period by class: DSCT 0.047 d, ECL 0.474 d, RR 0.568 d, CEP 3.94 d,
    LPV 296.8 d. Four orders of magnitude of period separating six classes, measured rather
    than assumed. This is the ladder P1 and P2 are about.

DERIVED QUANTITIES, FLAGGED AS SUCH
-----------------------------------
Two columns are computed rather than served, and both are falsifiable:

  * `magerr = 1.0857362 / flux_over_error`. Gaia publishes no per-epoch magnitude error;
    this is the standard first-order propagation of a flux error into magnitudes.
  * `mjd = time + 55197.0`. Epoch times are Barycentric JD in TCB minus 2455197.5 days.
    The summary reports the resulting MJD range; DR3 photometry spans roughly MJD 56863 to
    57902 (2014-07-25 to 2017-05-28), so an offset error would show up immediately as a
    range outside that window rather than passing silently.

The two safety mechanisms here -- the per-class retrieval-rate warning and the batch-halving
fallback -- are fault-injection tested by `src/verify_fetch_gaia_variables.py`, which drives
each with an injected failure and requires it to react. That check earned its place: it found
that the first draft of the halving fallback lost a healthy identifier alongside the bad one.

Run:  python3 src/fetch_gaia_variables.py [--per-class 500] [--workers 12] [--batch 25]
      python3 src/fetch_gaia_variables.py --per-class 50 --tag pilot   # pilot, ~8 min
      python3 src/verify_fetch_gaia_variables.py                       # 3/3 checks
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Six classes, all genuinely periodic, chosen so that period alone spans four orders of
# magnitude: DSCT hours, RR ~0.5 d, ECL hours to weeks, CEP days to months, RS days to
# weeks, LPV hundreds of days. This is the ladder that a period-blind representation should
# be unable to climb, which is why the counter-test uses it. AGN, SOLAR_LIKE, YSO and the
# rest of the DR3 taxonomy are excluded: they are stochastic or quasi-periodic, and their
# presence would soften exactly the contrast the experiment is built to measure.
CLASSES = ["ECL", "RR", "CEP", "RS", "DSCT|GDOR|SXPHE", "LPV"]

# Short aliases for filenames and printing; the pipe character in the DR3 label is awkward
# everywhere else.
ALIAS = {"ECL": "ECL", "RR": "RR", "CEP": "CEP", "RS": "RS",
         "DSCT|GDOR|SXPHE": "DSCT", "LPV": "LPV"}

# Where a catalogued period lives for each class, and in what units.
#
# A PREDICTION THAT FAILED, RECORDED HERE RATHER THAN QUIETLY DROPPED. The plan was to pull
# a catalogued period for every class so that a period-only baseline could be built by
# lookup. Measured overlap between the classifier table and the class-specific period
# tables says that only three of the six classes support it:
#
#   ECL  vari_eclipsing_binary       50/50 in the pilot; 2,184,477 rows, near-complete
#   RR   vari_rrlyrae                50/50 in the pilot
#   CEP  vari_cepheid                47/50 in the pilot
#   LPV  vari_long_period_variable   19/50; 299,648 of 710,604 LPV at score >= 0.8 carry a
#                                    non-null `frequency`, i.e. about 42 per cent
#   DSCT vari_ms_oscillator           5/50; 19,165 of 90,335 DSCT at score >= 0.8, ~21 per cent
#   RS   vari_rotation_modulation     0/50; only 994 of 63,477 RS at score >= 0.8 appear in
#                                    that table at all, about 1.6 per cent. The RS class and
#                                    the rotation-modulation catalogue are largely disjoint
#                                    populations in DR3.
#
# The consequence for T1: the period baseline for the counter-test must be a period the
# pipeline MEASURES from the epoch photometry (Lomb-Scargle or equivalent), not one it looks
# up. That is the better experiment in any case, since a catalogue period was derived with
# access to more information than the classifier is given, and importing it would flatter the
# baseline for the wrong reason. The catalogued period is retained only as an external
# cross-check on the measured one, on the three classes where it is available.
PERIOD_SOURCE = {
    "ECL": ("vari_eclipsing_binary", "frequency", "freq"),
    "RR": ("vari_rrlyrae", "fund_freq1", "freq"),
    "CEP": ("vari_cepheid", "fund_freq1", "freq"),
    "RS": ("vari_rotation_modulation", "best_rotation_period", "period"),
    "LPV": ("vari_long_period_variable", "frequency", "freq"),
    "DSCT|GDOR|SXPHE": ("vari_ms_oscillator", "frequency1", "freq"),
}

# Measured in the pilot; used to decide whether a low match rate is news or expected.
PERIOD_EXPECTED_RATE = {"ECL": 0.95, "RR": 0.95, "CEP": 0.90,
                        "LPV": 0.42, "DSCT|GDOR|SXPHE": 0.21, "RS": 0.02}

MIN_SCORE = 0.8          # classifier confidence floor; see docstring for the count at each
MIN_OBS_TOTAL = 30       # long-form observations, i.e. summed over the three bands
MIN_PER_BAND = 5         # each of G, BP, RP must be genuinely sampled
MAX_IDS_PER_CALL = 5000  # hard server limit, verified: 5000 -> 200, 5001 -> 500
BJD_TCB_TO_MJD = 55197.0

# Band -> (time column, mag column, flux column, flux error, flux/error, reject flag)
BANDS = {
    "G":  ("g_transit_time", "g_transit_mag", "g_transit_flux",
           "g_transit_flux_error", "g_transit_flux_over_error", "variability_flag_g_reject"),
    "BP": ("bp_obs_time", "bp_mag", "bp_flux",
           "bp_flux_error", "bp_flux_over_error", "variability_flag_bp_reject"),
    "RP": ("rp_obs_time", "rp_mag", "rp_flux",
           "rp_flux_error", "rp_flux_over_error", "variability_flag_rp_reject"),
}

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# --------------------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------------------
def fetch_labels(cache: Path, per_class: int, min_score: float,
                 oversample: float = 1.4) -> pd.DataFrame:
    """Pull a random per-class candidate pool from the DR3 variability classifier.

    One async TAP job per class. `ORDER BY s.random_index` is what makes the draw a uniform
    random subset of the class rather than a patch of sky; see the module docstring.

    More than `per_class` is requested so that the quality cuts have slack to remove
    objects without unbalancing the classes. The cache is the exact candidate pool, so the
    sample is reproducible even if the archive is re-released.
    """
    if cache.exists():
        df = pd.read_csv(cache)
        _log(f"candidate pool from cache: {cache} ({len(df)} rows)")
        return df

    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = -1

    want = int(np.ceil(per_class * oversample))
    frames = []
    for cls in CLASSES:
        q = (f"SELECT TOP {want} v.source_id, v.best_class_name, v.best_class_score, "
             f"s.random_index, s.phot_g_mean_mag "
             f"FROM gaiadr3.vari_classifier_result AS v "
             f"JOIN gaiadr3.gaia_source AS s ON s.source_id = v.source_id "
             f"WHERE v.best_class_name = '{cls}' AND v.best_class_score >= {min_score} "
             f"ORDER BY s.random_index")
        t0 = time.time()
        d = Gaia.launch_job_async(q).get_results().to_pandas()
        _log(f"  {ALIAS[cls]:5s} {len(d):5d} candidates in {time.time() - t0:.0f} s")
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    _log(f"candidate pool: {len(df)} rows -> {cache}")
    return df


def fetch_periods(source_ids: list[int], labels: dict[int, str], cache: Path,
                  chunk: int = 500) -> pd.DataFrame:
    """Pull the catalogued period for each source from its class-specific DR3 table.

    Non-fatal: a failure here costs the period-only baseline, not the sample. ADQL `IN`
    lists of 500 identifiers were measured working (10 kB query, 14 s, 491/500 matched for
    RR Lyrae), so the ids are chunked at that size.
    """
    if cache.exists():
        d = pd.read_csv(cache)
        _log(f"periods from cache: {cache} ({len(d)} rows)")
        return d

    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = -1

    by_class: dict[str, list[int]] = {}
    for sid in source_ids:
        by_class.setdefault(labels[sid], []).append(sid)

    rows = []
    for cls, ids in by_class.items():
        table, col, kind = PERIOD_SOURCE[cls]
        got = 0
        for i in range(0, len(ids), chunk):
            inlist = ",".join(str(s) for s in ids[i:i + chunk])
            q = (f"SELECT source_id, {col} AS raw_period_col "
                 f"FROM gaiadr3.{table} WHERE source_id IN ({inlist})")
            try:
                d = Gaia.launch_job_async(q).get_results().to_pandas()
            except Exception as exc:                       # noqa: BLE001
                _log(f"  period query failed for {ALIAS[cls]}: {type(exc).__name__}")
                continue
            v = pd.to_numeric(d["raw_period_col"], errors="coerce")
            period = np.where(v > 0, 1.0 / v, np.nan) if kind == "freq" else v.to_numpy()
            rows.append(pd.DataFrame({"source_id": d["source_id"].astype("int64"),
                                      "period_d": period}))
            got += int(np.isfinite(period).sum())
        rate = got / max(len(ids), 1)
        expected = PERIOD_EXPECTED_RATE.get(cls, 0.0)
        note = "" if rate >= 0.6 * expected else "  <- BELOW the measured expectation"
        _log(f"  {ALIAS[cls]:5s} periods: {got}/{len(ids)} ({rate:.0%}, expected ~{expected:.0%}) "
             f"from gaiadr3.{table}.{col}{note}")

    out = (pd.concat(rows, ignore_index=True) if rows
           else pd.DataFrame({"source_id": [], "period_d": []}))
    out.to_csv(cache, index=False)
    return out


def select_sample(pool: pd.DataFrame, per_class: int, seed: int = 20260730) -> pd.DataFrame:
    """Cap each class at `per_class`, then shuffle before any network work happens.

    The shuffle is not cosmetic. An earlier fetch in this project submitted objects grouped
    by class; the server's failure rate rose over the run, so the classes submitted last
    lost far more objects than those submitted first and one class was destroyed outright.
    That is a selection effect on the label, and it silently corrupts every downstream
    number. Shuffling makes any remaining attrition independent of class, and the per-class
    retrieval-rate check below is what proves it did.
    """
    pool = pool.copy()
    pool["best_class_name"] = pool["best_class_name"].astype(str).str.strip()
    before = len(pool)
    pool = pool[pool["best_class_name"].isin(CLASSES)]
    if len(pool) != before:
        _log(f"  dropped {before - len(pool)} rows outside the six retained classes")

    parts = [g.sample(min(len(g), per_class), random_state=seed)
             for _, g in pool.groupby("best_class_name", sort=True)]
    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    _log("class balance of the selected sample:")
    for cls, n in sample["best_class_name"].value_counts().items():
        _log(f"  {ALIAS[cls]:5s} {n:5d}")
    return sample


# --------------------------------------------------------------------------------------
# epoch photometry
# --------------------------------------------------------------------------------------
def _cache_paths(cache_dir: Path, sid: int) -> tuple[Path, Path]:
    return cache_dir / f"{sid}.parquet", cache_dir / f"{sid}.empty"


def _fetch_batch(ids: list[int], cache_dir: Path, retries: int = 4) -> tuple[int, int]:
    """Fetch one DATALINK batch and write one cache file per source. Returns (ok, failed).

    On repeated failure the batch is halved and retried, so one pathological identifier
    cannot cost the whole batch. Sources that come back genuinely empty get a `.empty`
    marker and are never retried on a later run.

    The recursion terminates on `len(ids) == 1` rather than on a depth cap. An earlier draft
    capped depth at four, and the fault-injection check found that on a 25-id batch this lost
    one healthy identifier alongside the poisoned one (23 recovered, not 24): halving reached
    a pair at the cap and abandoned both. Depth is bounded anyway at log2(5000) < 13.
    """
    if not ids:
        return 0, 0
    if len(ids) > MAX_IDS_PER_CALL:
        raise ValueError(f"batch of {len(ids)} exceeds the verified server cap of "
                         f"{MAX_IDS_PER_CALL}")

    from astroquery.gaia import Gaia

    for attempt in range(retries):
        try:
            products = Gaia.load_data(
                ids=ids, data_release="Gaia DR3", retrieval_type="EPOCH_PHOTOMETRY",
                data_structure="INDIVIDUAL", format="csv", verbose=False)
            frames = []
            for value in products.values():
                for tab in (value if isinstance(value, list) else [value]):
                    frames.append(tab.to_pandas())
            got = set()
            for df in frames:
                if df.empty or "source_id" not in df.columns:
                    continue
                for sid, grp in df.groupby("source_id"):
                    sid = int(sid)
                    grp.to_parquet(_cache_paths(cache_dir, sid)[0], index=False)
                    got.add(sid)
            for sid in ids:
                if sid not in got:
                    _cache_paths(cache_dir, sid)[1].write_text("")
            return len(got), len(ids) - len(got)
        except Exception as exc:                            # noqa: BLE001
            if attempt == retries - 1:
                if len(ids) > 1:
                    mid = len(ids) // 2
                    a = _fetch_batch(ids[:mid], cache_dir, retries=2)
                    b = _fetch_batch(ids[mid:], cache_dir, retries=2)
                    return a[0] + b[0], a[1] + b[1]
                _log(f"    source {ids[0]} failed permanently: "
                     f"{type(exc).__name__}: {str(exc)[:90]}")
                return 0, 1
            time.sleep(3.0 * (2 ** attempt))                # 3, 6, 12 s
    return 0, len(ids)


def fetch_epoch_photometry(sample: pd.DataFrame, workers: int, batch: int,
                           cache_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Retrieve wide-format epoch photometry for the sample, resuming from cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ids = [int(s) for s in sample["source_id"]]

    todo = [s for s in ids
            if not _cache_paths(cache_dir, s)[0].exists()
            and not _cache_paths(cache_dir, s)[1].exists()]
    _log(f"epoch photometry: {len(ids) - len(todo)}/{len(ids)} already cached, "
         f"{len(todo)} to fetch in batches of {batch} on {workers} workers")

    t0 = time.time()
    ok = bad = 0
    if todo:
        # astroquery writes its scratch directory into os.getcwd() and removes it in a
        # finally block; with a thread pool that litters whatever directory the script was
        # launched from, so the process is moved into the cache directory for the duration.
        # Every other path in this module is absolute, so this is safe.
        prev_cwd = Path.cwd()
        scratch = cache_dir / "_astroquery_scratch"
        scratch.mkdir(exist_ok=True)
        os.chdir(scratch)
        try:
            batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_fetch_batch, b, cache_dir): b for b in batches}
                for i, fut in enumerate(as_completed(futures), 1):
                    a, f = fut.result()
                    ok += a
                    bad += f
                    done = ok + bad
                    rate = done / max(time.time() - t0, 1e-9)
                    _log(f"  batch {i}/{len(batches)}: {done}/{len(todo)} sources "
                         f"({rate:.2f}/s, {bad} failed)")
        finally:
            os.chdir(prev_cwd)

    # Reload everything from cache, so a resumed run and a fresh run produce the same frame.
    frames = []
    for sid in ids:
        p, _ = _cache_paths(cache_dir, sid)
        if p.exists():
            frames.append(pd.read_parquet(p))
    wide = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    got = set(wide["source_id"].astype("int64").unique()) if not wide.empty else set()
    stats = {"requested": len(ids), "fetched_this_run": ok, "failed_this_run": bad,
             "sources_on_disk": len(got), "elapsed_s": round(time.time() - t0, 1),
             "sources_per_s": round(ok / max(time.time() - t0, 1e-9), 3) if ok else None,
             "transit_rows": int(len(wide))}
    _log(f"epoch photometry: {len(got)}/{len(ids)} sources, {len(wide)} transit rows, "
         f"{stats['elapsed_s']} s this run")

    # THE CHECK THAT EXISTS BECAUSE IT ONCE FAILED. Retrieval must not correlate with class.
    # This would fire if, say, LPV objects were systematically larger and timed out more
    # often, or if a batch failure happened to land on one class. It measures something:
    # give the fetcher an ordered rather than shuffled sample and the spread widens.
    rates = (sample.assign(ok=sample["source_id"].astype("int64").isin(got))
             .groupby("best_class_name")["ok"].agg(["sum", "count"]))
    rates["rate"] = (rates["sum"] / rates["count"]).round(3)
    _log("retrieval rate by class:")
    _log(rates.to_string())
    stats["retrieval_rate_by_class"] = rates["rate"].to_dict()
    spread = float(rates["rate"].max() - rates["rate"].min())
    stats["retrieval_rate_spread"] = round(spread, 3)
    if spread > 0.15:
        _log(f"  WARNING: retrieval rate varies by {spread:.2f} across classes; "
             f"attrition may be class-dependent and the labels may be biased")
    return wide, stats


# --------------------------------------------------------------------------------------
# reshape and cuts
# --------------------------------------------------------------------------------------
def reshape_to_long(wide: pd.DataFrame, labels: dict[int, str]) -> tuple[pd.DataFrame, dict]:
    """Wide (one row per transit, three timestamps) -> long (t, mag, err, band) tuples.

    Verified against the live service: the wide frame carries 25 columns, and
    `g_transit_time`, `bp_obs_time` and `rp_obs_time` differ within a single transit by
    of order 0.0004 d. The three bands really are sampled at different instants, so they
    are emitted as three separate observations rather than collapsed onto one timestamp.
    """
    wide = wide.copy()
    wide["source_id"] = wide["source_id"].astype("int64")
    parts, per_band_raw = [], {}

    for band, (tcol, mcol, fcol, fecol, focol, rej) in BANDS.items():
        cols = ["source_id", tcol, mcol, fcol, fecol, focol]
        missing = [c for c in cols if c not in wide.columns]
        if missing:
            raise KeyError(f"expected DATALINK columns absent for {band}: {missing}")
        d = wide[cols + ([rej] if rej in wide.columns else [])].copy()
        d.columns = (["source_id", "mjd", "mag", "flux", "fluxerr", "flux_over_error"]
                     + (["reject"] if rej in wide.columns else []))
        per_band_raw[band] = int(len(d))
        d["band"] = band
        parts.append(d)

    long = pd.concat(parts, ignore_index=True)
    n0 = len(long)

    for c in ("mjd", "mag", "flux", "fluxerr", "flux_over_error"):
        long[c] = pd.to_numeric(long[c], errors="coerce")

    # Every removal is counted. A silent drop here is indistinguishable from a bug.
    finite = long["mjd"].notna() & long["mag"].notna() & long["flux"].notna()
    n_nonfinite = int((~finite).sum())
    long = long[finite]

    if "reject" in long.columns:
        flag = long["reject"].astype(str).str.lower().isin(["true", "1", "t"])
        n_rejected = int(flag.sum())
        long = long[~flag].drop(columns=["reject"])
    else:
        n_rejected = 0

    pos = long["flux_over_error"] > 0
    n_bad_snr = int((~pos).sum())
    long = long[pos]

    long["magerr"] = 1.0857362 / long["flux_over_error"]
    long["mjd"] = long["mjd"] + BJD_TCB_TO_MJD
    long["label"] = long["source_id"].map(labels)
    long = (long[["source_id", "label", "mjd", "band", "mag", "magerr",
                  "flux", "fluxerr"]]
            .sort_values(["source_id", "mjd"]).reset_index(drop=True))

    info = {"wide_transit_rows": int(len(wide)),
            "long_slots_before_cuts": n0,
            "dropped_missing_photometry": n_nonfinite,
            "dropped_variability_reject_flag": n_rejected,
            "dropped_nonpositive_snr": n_bad_snr,
            "long_observations": int(len(long)),
            "observations_by_band": long["band"].value_counts().to_dict(),
            "mjd_min": round(float(long["mjd"].min()), 2),
            "mjd_max": round(float(long["mjd"].max()), 2)}
    _log(f"reshape: {info['wide_transit_rows']} transits -> {n0} band-slots -> "
         f"{info['long_observations']} observations "
         f"({n_nonfinite} unobserved, {n_rejected} reject-flagged, {n_bad_snr} bad SNR)")

    # Falsifiable epoch check: DR3 photometry spans MJD ~56863 to ~57902. If the TCB offset
    # were wrong this fires instead of quietly shifting every timestamp.
    if not (56800 < info["mjd_min"] < 57950 and 56800 < info["mjd_max"] < 57950):
        _log(f"  WARNING: MJD range {info['mjd_min']}..{info['mjd_max']} lies outside the "
             f"Gaia DR3 window (~56863..57902); check BJD_TCB_TO_MJD")
        info["mjd_range_check"] = "FAILED"
    else:
        info["mjd_range_check"] = "passed"
        _log(f"  MJD range {info['mjd_min']}..{info['mjd_max']} — inside the DR3 window")
    return long, info


def apply_quality_cuts(long: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop light curves too sparse to carry shape. Each cut reports what it removed."""
    before = int(long["source_id"].nunique())

    counts = long.groupby("source_id").size()
    enough = counts[counts >= MIN_OBS_TOTAL].index
    long = long[long["source_id"].isin(enough)]

    per_band = long.groupby(["source_id", "band"]).size().unstack(fill_value=0)
    for b in BANDS:
        if b not in per_band.columns:
            per_band[b] = 0
    keep = per_band[(per_band["G"] >= MIN_PER_BAND)
                    & (per_band["BP"] >= MIN_PER_BAND)
                    & (per_band["RP"] >= MIN_PER_BAND)].index
    long = long[long["source_id"].isin(keep)]

    long = long.sort_values(["source_id", "mjd"]).reset_index(drop=True)
    cuts = {"objects_before": before,
            "objects_after": int(long["source_id"].nunique()),
            "dropped_too_few_observations": int(before - len(enough)),
            "dropped_missing_a_band": int(len(enough) - len(keep)),
            "observations_after": int(len(long))}
    _log(f"quality cuts: {cuts['objects_before']} -> {cuts['objects_after']} objects "
         f"({cuts['dropped_too_few_observations']} too sparse, "
         f"{cuts['dropped_missing_a_band']} missing a band)")
    return long, cuts


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=500,
                    help="objects per class before quality cuts")
    ap.add_argument("--workers", type=int, default=12,
                    help="concurrent DATALINK connections; the stream is throttled per "
                         "connection, so this is the throughput knob")
    ap.add_argument("--batch", type=int, default=25,
                    help=f"source ids per DATALINK call (server cap {MAX_IDS_PER_CALL})")
    ap.add_argument("--min-score", type=float, default=MIN_SCORE)
    ap.add_argument("--tag", default="", help="suffix for output filenames, e.g. 'pilot'")
    ap.add_argument("--no-periods", action="store_true",
                    help="skip the catalogued-period pull")
    args = ap.parse_args()

    if args.batch > MAX_IDS_PER_CALL:
        print(f"--batch must not exceed {MAX_IDS_PER_CALL}", file=sys.stderr)
        return 2

    suffix = f"_{args.tag}" if args.tag else ""
    data = ROOT / "data"
    outputs = ROOT / "outputs"
    data.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)

    pool = fetch_labels(data / f"gaia_dr3_vari_pool{suffix}.csv",
                        args.per_class, args.min_score)
    sample = select_sample(pool, args.per_class)
    labels = {int(s): c for s, c in zip(sample["source_id"], sample["best_class_name"])}

    wide, fetch_stats = fetch_epoch_photometry(
        sample, args.workers, args.batch, data / "gaia_epoch_cache")
    if wide.empty:
        print("no epoch photometry retrieved", file=sys.stderr)
        return 1

    long, reshape_info = reshape_to_long(wide, labels)
    long, cuts = apply_quality_cuts(long)
    long.to_parquet(data / f"gaia_dr3_variables{suffix}.parquet", index=False)

    manifest = (long.groupby("source_id")
                .agg(label=("label", "first"), n_obs=("mjd", "size"),
                     t_span=("mjd", lambda s: round(s.max() - s.min(), 3)))
                .reset_index())

    period_stats: dict = {"requested": False}
    if not args.no_periods:
        per = fetch_periods([int(s) for s in manifest["source_id"]], labels,
                            data / f"gaia_dr3_periods{suffix}.csv")
        if not per.empty:
            manifest = manifest.merge(per, on="source_id", how="left")
            got = manifest.groupby("label")["period_d"].apply(lambda s: s.notna().mean())
            med = manifest.groupby("label")["period_d"].median().round(4)
            # None rather than NaN: json.dumps emits a bare NaN token, which Python reads
            # back but strict JSON parsers reject, so the summary would be unusable
            # elsewhere. RS is expected to be null here -- see PERIOD_SOURCE.
            period_stats = {"requested": True,
                            "match_rate_by_class": got.round(3).to_dict(),
                            "median_period_d_by_class":
                                {k: (None if pd.isna(v) else float(v))
                                 for k, v in med.items()}}
            _log("catalogued period, median days by class:")
            for cls, v in period_stats["median_period_d_by_class"].items():
                rate = period_stats["match_rate_by_class"].get(cls, 0.0)
                _log(f"  {ALIAS.get(cls, cls):5s} {'unavailable' if v is None else v}"
                     f"   ({rate:.0%} of objects)")
        else:
            period_stats = {"requested": True, "error": "no periods returned"}

    manifest.to_csv(outputs / f"gaia_sample_manifest{suffix}.csv", index=False)

    obs = long.groupby("source_id").size()
    summary = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "classes": CLASSES,
        "per_class_cap": args.per_class,
        "min_class_score": args.min_score,
        "min_obs_total": MIN_OBS_TOTAL,
        "min_per_band": MIN_PER_BAND,
        "datalink": {"data_structure": "INDIVIDUAL", "format": "csv",
                     "ids_per_call": args.batch, "workers": args.workers,
                     "verified_id_cap": MAX_IDS_PER_CALL},
        "fetch": fetch_stats,
        "reshape": reshape_info,
        "quality_cuts": cuts,
        "periods": period_stats,
        "class_counts": manifest["label"].value_counts().to_dict(),
        "median_observations": int(obs.median()),
        "observations_p10_p90": [int(obs.quantile(0.10)), int(obs.quantile(0.90))],
        "median_span_days": round(float(manifest["t_span"].median()), 1),
    }
    (outputs / f"gaia_fetch_summary{suffix}.json").write_text(json.dumps(summary, indent=2))
    _log(json.dumps(summary["class_counts"], indent=2))
    _log(f"median observations per object: {summary['median_observations']} "
         f"(p10-p90 {summary['observations_p10_p90']}), "
         f"median span {summary['median_span_days']} d")
    _log(f"wrote {data / f'gaia_dr3_variables{suffix}.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
