"""The T1 counter-test: do signatures fail on Gaia DR3 periodic variables, as predicted?

WHAT THIS MODULE IS FOR
-----------------------
T1 measured signature features against a hand-crafted variability baseline on ZTF BTS
transients, where the discriminant is the *shape and ordering* of the light curve. On that
sample signatures came close: 0.548 balanced accuracy against 0.560 for the baseline, a gap
of about 0.015 on paired folds, and the signature block was complementary to the baseline
rather than redundant with it.

That result is only interesting if the stated mechanism is real. The claim was that
signatures win where ordering carries the signal. The sharpest way to test a claim of that
form is to find a sample where the mechanism says the method must LOSE, and check that it
does. Gaia DR3 periodic variables are that sample: an eclipsing binary, an RR Lyrae, a
Cepheid, a spotted rotator, a delta-Scuti pulsator and a long-period variable are separated
chiefly by PERIOD, and period is a statement about the clock, not about the shape. The
signature is invariant to increasing reparameterisation of the clock, so a plain signature
should discard exactly the discriminant that matters here.

Two predictions were recorded in the docstring of `src/fetch_gaia_variables.py` BEFORE any
of this was run, so that either could fail on the record:

  P1. Signatures should lose to the hand-crafted baseline by a markedly LARGER margin on
      Gaia than the ~0.015 measured on ZTF.
  P2. An explicit time channel should be worth MORE on Gaia than the +0.069 it recovered on
      ZTF, because here the clock is the signal rather than the weather.

If signatures instead win here too, for the reasons claimed on ZTF, then the account of the
mechanism given in T1 is wrong and the track's headline has to change. A method that wins
everywhere for the same stated reason has not been understood.

WHAT IS HELD FIXED, AND WHY THAT IS THE WHOLE POINT
---------------------------------------------------
A cross-sample comparison is worthless unless the machinery either side of the data is
identical. Nothing is reimplemented here. The three components that decide the numbers are
imported from the modules that produced the ZTF result:

  features_baseline.feature_frame        the hand-crafted variability features
  features_signature.signature_feature_frame   the path construction and the signature
  run_benchmark.evaluate, run_benchmark._align  the evaluation protocol and the folds
  test_complementarity.paired_scores, .compare, .bootstrap_ci   the paired-fold design

`evaluate` carries its own `StratifiedKFold(random_state=SEED)`, so the fold construction is
byte-identical between the two tracks; only the data differ.

The ZTF reference numbers are RE-MEASURED here rather than copied from the earlier output
files, for one specific reason: the published ZTF benchmark never recorded a `unit_time`
variant, so the +0.069 time-channel figure quoted in the project README could not be
recovered from `outputs/benchmark_results.json`. Rather than retype a number, this module
recomputes both tracks' time-channel deltas with the same code in the same process. The
previously published ZTF figures are still loaded and reported alongside as a consistency
check on that re-measurement; a mismatch there would mean something drifted.

THREE DIFFERENCES BETWEEN THE SAMPLES, STATED RATHER THAN BURIED
---------------------------------------------------------------
  1. Gaia has three bands (G, BP, RP), ZTF two (g, r). The band tuple is passed through, so
     the per-band signature yields three paths instead of two and the baseline yields three
     blocks instead of two.
  2. `features_baseline.object_features` emits exactly ONE colour term, between `bands[0]`
     and `bands[1]`, whatever the number of bands. On Gaia that gives G-BP and silently
     omits G-RP and BP-RP. This mildly handicaps the baseline on Gaia. It is left alone:
     editing the shared module would invalidate the ZTF result that is the comparison.
  3. The Gaia sample is denser and far longer-baselined than the ZTF sample (median ~127
     observations over ~960 d against ZTF's sparse, short curves). Absolute balanced
     accuracies are therefore NOT comparable across the two tracks, and no claim here rests
     on comparing them. Every reported quantity is a within-sample margin: baseline minus
     signature, or one preparation minus another, measured on the same folds and the same
     objects. That is the only form of comparison the two samples support.

CHECKS THAT CAN FAIL
--------------------
Three, all cheap, all reported in the output JSON:

  * Label-shuffle control. The baseline block is re-evaluated with permuted labels. With six
    balanced classes it must land near 1/6 = 0.167. Anything materially above that means the
    folds or the label join leak, and every other number in the file would be void. This
    check fails loudly rather than silently, which is the only reason to run it.
  * Increment-invariance check. `raw` and `raw_time` differ only by a constant magnitude
    offset per object, and a signature depends only on increments, so the two MUST score
    identically. On ZTF they agreed to four decimal places. If they disagree on Gaia, the
    path construction is not doing what the theory says and the depth ablation is unsafe.
  * Matched-noise control in the complementarity stage. The signature block is replaced by
    Gaussian noise of identical shape and per-column variance. If noise also appears to help,
    the apparent gain is classifier capacity rather than information, and the complementarity
    claim is void on this sample.

Run:  python3 src/run_gaia_countertest.py                       # everything, ~1.5 h
      python3 src/run_gaia_countertest.py --stages benchmark    # representations only
      python3 src/run_gaia_countertest.py --stages complementarity
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features_baseline import feature_frame  # noqa: E402
from features_signature import signature_feature_frame  # noqa: E402
from run_benchmark import _align, evaluate, interpolated_panel  # noqa: E402
from test_complementarity import bootstrap_ci, compare, paired_scores  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260726          # the ZTF seed, deliberately unchanged

GAIA_BANDS = ("G", "BP", "RP")
ZTF_BANDS = ("g", "r")

# The variant grid. Deliberately the same shape as the ZTF ablation reported in the project
# README, so that "the same representations" is literally true rather than approximately so.
# Restricted to per_band because that is the construction the published ZTF benchmark used;
# it is also the only one that involves no interpolation of any kind.
VARIANTS = [
    # (depth, norm, augmentation)
    (3, "unit_time", "none"),
    (4, "unit_time", "none"),
    (3, "raw_time", "none"),
    (4, "raw_time", "none"),
    (4, "raw", "none"),              # increment-invariance check against raw_time
    (4, "raw_time", "basepoint"),
    (3, "unit_time", "lead_lag"),
    (4, "unit_time", "lead_lag"),
    (3, "raw_time", "lead_lag"),
    (4, "raw_time", "lead_lag"),
]

# The configuration the ZTF complementarity test used, reused verbatim.
COMPLEMENTARITY_CONFIG = {"mode": "per_band", "norm": "raw_time", "depth": 4,
                          "lead_lag": True}

# Baseline features that a plain signature cannot represent even in principle, because a
# signature depends only on the increments of the path: adding a constant to every magnitude
# of an object leaves it unchanged. Everything else in the baseline -- amplitude, dispersion,
# skew, Stetson K, slopes, cadence, span, rise and fade fractions -- is offset-invariant and
# therefore inside the signature's reach in principle.
#
# This matters for reading the headline gap honestly. If the baseline wins mainly through the
# absolute brightness and colour of the star, the gap is a statement about which CHANNELS the
# representation is handed, not about ordering, and the counter-test would be measuring the
# wrong thing. Dropping these columns and re-scoring separates the two accounts.
_OFFSET_DEPENDENT = ("mean", "median", "peak_mag", "percent_amplitude")


def variant_key(mode: str, depth: int, norm: str, aug: str) -> str:
    """Name a variant exactly as run_benchmark.main does, so keys line up across tracks."""
    suffix = "" if norm == "unit_time" else f"-{norm}"
    suffix += "" if aug == "none" else f"-{aug}"
    return f"signature-{mode}{suffix}-d{depth}"


def load_gaia(tag: str) -> pd.DataFrame:
    """Load the Gaia long-format photometry and rename its object key to `oid`.

    The shared feature modules group by `oid`; the fetcher emits `source_id`. Renaming here
    keeps both untouched. Nothing else about the frame is altered -- no binning, no
    interpolation, no resampling.
    """
    suffix = f"_{tag}" if tag else ""
    path = ROOT / "data" / f"gaia_dr3_variables{suffix}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run src/fetch_gaia_variables.py --per-class 500 first")
    df = pd.read_parquet(path).rename(columns={"source_id": "oid"})
    missing = {"oid", "label", "mjd", "band", "mag", "magerr"} - set(df.columns)
    if missing:
        raise KeyError(f"{path} lacks required columns {sorted(missing)}")
    seen = set(df["band"].unique())
    if seen != set(GAIA_BANDS):
        raise ValueError(f"expected bands {set(GAIA_BANDS)} in {path}, found {seen}")
    return df


def load_ztf() -> pd.DataFrame:
    path = ROOT / "data" / "ztf_bts_lightcurves.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run src/fetch_ztf_bts.py first")
    return pd.read_parquet(path)


def describe(df: pd.DataFrame, name: str) -> dict:
    """Sample description computed from the photometry itself, not from a summary file."""
    per_obj = df.groupby("oid").agg(label=("label", "first"), n_obs=("mjd", "size"),
                                    span=("mjd", lambda s: s.max() - s.min()))
    obs = per_obj["n_obs"]
    info = {
        "n_objects": int(len(per_obj)),
        "n_observations": int(len(df)),
        "n_classes": int(per_obj["label"].nunique()),
        "class_counts": per_obj["label"].value_counts().sort_index().to_dict(),
        "median_observations": float(obs.median()),
        "observations_p10_p90": [float(obs.quantile(0.10)), float(obs.quantile(0.90))],
        "median_span_days": round(float(per_obj["span"].median()), 2),
        "span_p10_p90_days": [round(float(per_obj["span"].quantile(0.10)), 2),
                              round(float(per_obj["span"].quantile(0.90)), 2)],
        # How much the observed BASELINE varies between objects. This is the quantity that
        # decides whether `unit_time` normalisation can destroy anything: rescaling each
        # object's time axis to [0, 1] removes duration, and duration can only be a
        # discriminant if it differs between objects. It is recorded because it turns out to
        # be the reason prediction P2 behaves as it does.
        "span_coefficient_of_variation": round(
            float(per_obj["span"].std() / per_obj["span"].mean()), 4),
        "span_ratio_p90_over_p10": round(
            float(per_obj["span"].quantile(0.90) / max(per_obj["span"].quantile(0.10),
                                                       1e-9)), 3),
        "observations_by_band": df["band"].value_counts().sort_index().to_dict(),
        "chance_balanced_accuracy": round(1.0 / per_obj["label"].nunique(), 4),
    }
    print(f"\n{name}: {info['n_objects']} objects, {info['n_observations']} observations, "
          f"{info['n_classes']} classes")
    print(f"  median {info['median_observations']:.0f} observations per object "
          f"(p10-p90 {info['observations_p10_p90']}), "
          f"median span {info['median_span_days']} d")
    print("  " + "  ".join(f"{k}={v}" for k, v in info["class_counts"].items()))
    return info


# --------------------------------------------------------------------------- stage 1
def run_variant_grid(df: pd.DataFrame, bands: tuple[str, ...], folds: int,
                     mode: str, label: str, depths: set[int] | None = None) -> dict:
    """Evaluate the baseline and every signature variant on identical folds."""
    results: dict[str, dict] = {}

    print(f"\n[{label}] hand-crafted baseline ...")
    t0 = time.time()
    summary = feature_frame(df, bands=bands)
    y = summary["label"]
    results["summary"] = evaluate(summary.drop(columns=["label"]), y, folds)
    results["summary"]["build_seconds"] = round(time.time() - t0, 1)
    _show("summary", results["summary"])

    # Label-shuffle control on the baseline block. Must collapse to chance; if it does not,
    # the folds or the label join leak and nothing below can be trusted.
    rng = np.random.default_rng(SEED)
    y_shuf = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
    shuffled = evaluate(summary.drop(columns=["label"]), y_shuf, folds)
    results["summary-label-shuffled"] = shuffled
    chance = 1.0 / y.nunique()
    ok = abs(shuffled["boosted_trees"]["balanced_accuracy"] - chance) < 0.05
    print(f"  label-shuffle control: {shuffled['boosted_trees']['balanced_accuracy']:.4f} "
          f"against chance {chance:.4f} -> {'passed' if ok else 'FAILED'}")

    # The baseline stripped of everything a signature cannot see by construction.
    X = summary.drop(columns=["label"])
    drop = [c for c in X.columns
            if any(c == f"{b}_{f}" for b in bands for f in _OFFSET_DEPENDENT)
            or c.startswith("colour_")]
    results["summary-offset-invariant-only"] = evaluate(X.drop(columns=drop), y, folds)
    results["summary-offset-invariant-only"]["dropped_columns"] = drop
    results["summary-offset-invariant-only"]["note"] = (
        "the hand-crafted baseline with the absolute-brightness and colour features removed, "
        "i.e. restricted to quantities a plain increment-based signature could represent")
    _show("summary-offset-invariant-only", results["summary-offset-invariant-only"])

    best_key, best_frame, best_score = None, None, -1.0
    for depth, norm, aug in VARIANTS:
        if depths is not None and depth not in depths:
            continue
        key = variant_key(mode, depth, norm, aug)
        t0 = time.time()
        sig = signature_feature_frame(
            df, depth=depth, mode=mode, bands=bands, norm=norm,
            basepoint=(aug == "basepoint"), use_lead_lag=(aug == "lead_lag"))
        (Xs,), ys = _align(sig)
        results[key] = evaluate(Xs, ys, folds)
        results[key]["build_seconds"] = round(time.time() - t0, 1)
        _show(key, results[key])
        score = results[key]["boosted_trees"]["balanced_accuracy"]
        if score > best_score:
            best_key, best_frame, best_score = key, sig, score

    results["_best_signature"] = {"key": best_key, "balanced_accuracy": best_score}

    if best_frame is not None:
        print(f"\n[{label}] combined: summary + {best_key} ...")
        (Xa, Xb), yc = _align(summary, best_frame)
        combined = pd.concat([Xa, Xb], axis=1)
        results[f"summary+{best_key}"] = evaluate(combined, yc, folds)
        results[f"summary+{best_key}"]["note"] = (
            "concatenation of the hand-crafted features with the best signature variant")
        _show(f"summary+{best_key}", results[f"summary+{best_key}"])

    return results


def add_minirocket(df: pd.DataFrame, bands: tuple[str, ...], folds: int,
                   results: dict, label: str) -> None:
    """MiniRocket on a linearly interpolated grid: the baseline given its best case.

    Reuses run_benchmark.interpolated_panel so the interpolation is the same one the ZTF
    benchmark used. Skipped without complaint if sktime is unavailable, since it is a
    supporting number rather than one any conclusion rests on.
    """
    try:
        from sktime.transformations.panel.rocket import MiniRocketMultivariate
    except ImportError:
        print(f"[{label}] sktime unavailable, skipping MiniRocket")
        return
    print(f"\n[{label}] MiniRocket on interpolated grid ...")
    t0 = time.time()
    panel, oids, labels = interpolated_panel(df, bands=bands)
    tr = MiniRocketMultivariate(random_state=SEED)
    feats = np.asarray(tr.fit_transform(panel))
    mr = pd.DataFrame(feats, index=pd.Index(oids, name="oid"),
                      columns=[f"mr{i}" for i in range(feats.shape[1])])
    mr["label"] = labels
    (Xm,), ym = _align(mr)
    results["minirocket"] = evaluate(Xm, ym, folds)
    results["minirocket"]["build_seconds"] = round(time.time() - t0, 1)
    _show("minirocket", results["minirocket"])


def derive_headline(results: dict, mode: str) -> dict:
    """The two margins the counter-test is actually about, plus the invariance check.

    Both are within-sample differences on identical folds, which is the only comparison the
    two samples support.
    """
    bt = lambda k: results[k]["boosted_trees"]["balanced_accuracy"]  # noqa: E731

    baseline = bt("summary")
    best_key = results["_best_signature"]["key"]
    best = results["_best_signature"]["balanced_accuracy"]

    plain_key = variant_key(mode, 4, "raw_time", "none")
    unit_key = variant_key(mode, 4, "unit_time", "none")
    raw_key = variant_key(mode, 4, "raw", "none")

    # The headline time-channel figure is the depth-4, no-augmentation cell, because that is
    # the pair the ZTF +0.069 was read from. One cell is a thin basis for a claim, so the
    # same difference is also computed for every other (depth, augmentation) pair present.
    # If the headline cell disagrees in sign with the rest, that is reported rather than
    # hidden behind the convenient number.
    pairs = {}
    for depth, _, aug in VARIANTS:
        rk, uk = variant_key(mode, depth, "raw_time", aug), variant_key(
            mode, depth, "unit_time", aug)
        name = f"d{depth}-{aug}"
        if rk in results and uk in results and name not in pairs:
            pairs[name] = round(bt(rk) - bt(uk), 4)

    out = {
        "baseline_balanced_accuracy": baseline,
        "best_signature_key": best_key,
        "best_signature_balanced_accuracy": best,
        "gap_baseline_minus_best_signature": round(baseline - best, 4),
        "time_channel": {
            "unit_time_key": unit_key,
            "unit_time_balanced_accuracy": bt(unit_key),
            "raw_time_key": plain_key,
            "raw_time_balanced_accuracy": bt(plain_key),
            "value_of_keeping_duration": round(bt(plain_key) - bt(unit_key), 4),
            "all_pairs": pairs,
            "all_pairs_mean": round(float(np.mean(list(pairs.values()))), 4) if pairs
                              else None,
            "all_pairs_max": round(float(np.max(list(pairs.values()))), 4) if pairs
                             else None,
        },
    }

    # The other augmentation knob, measured for contrast. On ZTF the lead-lag transform was
    # worth about +0.025. It is reported here because if the time-scaling knob turns out to
    # be worth nothing on Gaia, the question of which knob DOES carry the clock information
    # becomes the interesting one, and it should be answered with a number rather than a
    # story.
    if "summary-offset-invariant-only" in results:
        out["baseline_without_absolute_photometry"] = {
            "balanced_accuracy": bt("summary-offset-invariant-only"),
            "cost_of_removing_it": round(
                baseline - bt("summary-offset-invariant-only"), 4),
            "gap_against_best_signature": round(
                bt("summary-offset-invariant-only") - best, 4),
            "what_it_tests": ("whether the baseline's advantage is ordering information or "
                              "absolute photometry the signature is blind to by construction"),
        }

    ll_key = variant_key(mode, 4, "raw_time", "lead_lag")
    if ll_key in results and plain_key in results:
        out["value_of_lead_lag"] = {
            "with_lead_lag_key": ll_key,
            "with_lead_lag_balanced_accuracy": bt(ll_key),
            "without_key": plain_key,
            "without_balanced_accuracy": bt(plain_key),
            "value": round(bt(ll_key) - bt(plain_key), 4),
        }
    if raw_key in results:
        d = abs(bt(raw_key) - bt(plain_key))
        out["increment_invariance_check"] = {
            "raw_balanced_accuracy": bt(raw_key),
            "raw_time_balanced_accuracy": bt(plain_key),
            "absolute_difference": round(d, 4),
            "verdict": "passed" if d < 5e-4 else "FAILED",
            "what_it_tests": ("a signature depends only on increments, so centring the "
                              "magnitudes cannot change it; a non-zero difference here "
                              "would mean the path construction is wrong"),
        }
    return out


# --------------------------------------------------------------------------- stage 2
def run_complementarity(df: pd.DataFrame, bands: tuple[str, ...], splits: int,
                        repeats: int, label: str) -> dict:
    """Paired-fold complementarity with a matched-noise control, as on ZTF.

    The noise block is Gaussian, of identical shape, scaled to the signature block's
    per-column standard deviation, so it differs from the real block only in information
    content. Constructed with the same formula and seed as test_complementarity.main.
    """
    cfg = COMPLEMENTARITY_CONFIG
    print(f"\n[{label}] complementarity: {cfg} ...")
    summary = feature_frame(df, bands=bands)
    sig = signature_feature_frame(df, depth=cfg["depth"], mode=cfg["mode"], bands=bands,
                                  norm=cfg["norm"], use_lead_lag=cfg["lead_lag"])
    common = summary.index.intersection(sig.index).sort_values()
    y = summary.loc[common, "label"].to_numpy()
    Xs = np.nan_to_num(summary.loc[common].drop(columns=["label"]).to_numpy(float))
    Xg = np.nan_to_num(sig.loc[common].drop(columns=["label"]).to_numpy(float))
    print(f"  {len(common)} objects, summary {Xs.shape[1]} features, "
          f"signature {Xg.shape[1]} features")

    rng = np.random.default_rng(SEED)
    Xn = rng.normal(0.0, Xg.std(axis=0, keepdims=True) + 1e-9, size=Xg.shape)

    blocks = {"summary": Xs,
              "summary+signature": np.hstack([Xs, Xg]),
              "summary+noise": np.hstack([Xs, Xn]),
              "signature": Xg}
    n_folds = splits * repeats
    print(f"  {n_folds} paired folds over {len(blocks)} blocks ...")
    t0 = time.time()
    scores = paired_scores(blocks, y, splits, repeats)
    for name, s in scores.items():
        print(f"    {name:20s} {s.mean():.4f} +- {s.std():.4f}")

    real = compare(scores, "summary", "summary+signature")
    control = compare(scores, "summary", "summary+noise")
    alone = compare(scores, "summary", "signature")
    verdict = {
        "signature_adds_over_baseline": bool(real["ci95"][0] > 0),
        "noise_control_clean": bool(control["ci95"][0] <= 0),
        "gain_exceeds_noise_control": bool(
            real["mean_difference"] > control["mean_difference"]),
        "complementary": bool(real["ci95"][0] > 0 and control["ci95"][0] <= 0),
    }
    print(f"  signature gain {real['mean_difference']:+.4f} "
          f"CI [{real['ci95'][0]:+.4f}, {real['ci95'][1]:+.4f}] p={real['wilcoxon_p']}")
    print(f"  noise control  {control['mean_difference']:+.4f} "
          f"CI [{control['ci95'][0]:+.4f}, {control['ci95'][1]:+.4f}]")
    print(f"  complementary: {verdict['complementary']}")

    return {
        "n_objects": int(len(common)),
        "splits": splits, "repeats": repeats,
        "signature_config": cfg,
        "n_summary_features": int(Xs.shape[1]),
        "n_signature_features": int(Xg.shape[1]),
        "elapsed_s": round(time.time() - t0, 1),
        "per_block_mean": {k: round(float(v.mean()), 4) for k, v in scores.items()},
        "per_block_std": {k: round(float(v.std()), 4) for k, v in scores.items()},
        "summary_vs_summary_plus_signature": real,
        "summary_vs_summary_plus_noise": control,
        "summary_vs_signature_alone": alone,
        # The paired gap is the quantity the ZTF +0.015 refers to, so it is named explicitly
        # rather than left to be read off a sign convention.
        "paired_gap_baseline_minus_signature": round(-alone["mean_difference"], 4),
        "verdict": verdict,
        "_fold_scores": {k: [round(float(x), 6) for x in v] for k, v in scores.items()},
    }


# --------------------------------------------------------------------------- reference
def published_ztf_reference() -> dict:
    """The ZTF numbers already in outputs/, for consistency-checking the re-measurement."""
    out: dict = {}
    b = ROOT / "outputs" / "benchmark_results.json"
    c = ROOT / "outputs" / "complementarity.json"
    if b.exists():
        r = json.loads(b.read_text())["results"]
        sig = {k: v["boosted_trees"]["balanced_accuracy"]
               for k, v in r.items() if k.startswith("signature-")}
        best = max(sig, key=sig.get)
        out["benchmark"] = {
            "summary": r["summary"]["boosted_trees"]["balanced_accuracy"],
            "best_signature_key": best,
            "best_signature": sig[best],
            "gap_baseline_minus_best_signature":
                round(r["summary"]["boosted_trees"]["balanced_accuracy"] - sig[best], 4),
        }
    if c.exists():
        d = json.loads(c.read_text())
        out["complementarity"] = {
            "paired_gap_baseline_minus_signature":
                round(-d["summary_vs_signature_alone"]["mean_difference"], 4),
            "signature_gain": d["summary_vs_summary_plus_signature"]["mean_difference"],
            "noise_cost": d["summary_vs_summary_plus_noise"]["mean_difference"],
        }
    return out


def ztf_reference_grid(ztf: pd.DataFrame, folds: int, mode: str) -> dict:
    """The ZTF variant grid, cached on disk so a re-run of the counter-test is cheap.

    The cache is fingerprinted on the variant grid, the mode, the fold count and the seed. If
    any of those change the cache is discarded and the grid is recomputed, so it cannot serve
    a stale answer to a different question -- which is the only way a cache like this can do
    damage.
    """
    fingerprint = {"variants": [list(v) for v in VARIANTS], "mode": mode,
                   "folds": folds, "seed": SEED, "bands": list(ZTF_BANDS)}
    cache = ROOT / "data" / "ztf_reference_representations.json"
    if cache.exists():
        blob = json.loads(cache.read_text())
        if blob.get("fingerprint") == fingerprint:
            print(f"\n[ztf] representation grid from cache: {cache}")
            return blob["representations"]
        print(f"\n[ztf] cache fingerprint changed, recomputing: {cache}")
    res = run_variant_grid(ztf, ZTF_BANDS, folds, mode, "ztf")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"fingerprint": fingerprint,
                                 "generated_utc": pd.Timestamp.utcnow().isoformat(),
                                 "representations": res}, indent=2))
    return res


def _show(name: str, r: dict) -> None:
    bt, lg = r["boosted_trees"], r["logistic"]
    print(f"  {name:42s} dim={r['n_features']:5d}  "
          f"trees={bt['balanced_accuracy']:.4f}+-{bt['balanced_accuracy_std']:.4f}  "
          f"logistic={lg['balanced_accuracy']:.4f}")


def _dump(payload: dict, out: Path) -> None:
    out.mkdir(exist_ok=True)
    (out / "gaia_countertest.json").write_text(json.dumps(payload, indent=2))


def write_table(payload: dict, out: Path) -> pd.DataFrame:
    """Flat CSV of every representation on both tracks, shaped like benchmark_table.csv."""
    rows = []
    for track in ("gaia", "ztf"):
        res = payload.get(track, {}).get("representations")
        if not res:
            continue
        for key, v in res.items():
            if key.startswith("_") or "boosted_trees" not in v:
                continue
            rows.append({
                "track": track,
                "representation": key,
                "n_features": v["n_features"],
                "n_objects": v["n_objects"],
                "balanced_accuracy": v["boosted_trees"]["balanced_accuracy"],
                "balanced_accuracy_std": v["boosted_trees"]["balanced_accuracy_std"],
                "macro_f1": v["boosted_trees"]["macro_f1"],
                "logistic_balanced_accuracy": v["logistic"]["balanced_accuracy"],
                "build_seconds": v.get("build_seconds"),
            })
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["track", "balanced_accuracy"], ascending=[True, False])
        table.to_csv(out / "gaia_countertest_table.csv", index=False)
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="suffix of the fetched parquet, e.g. 'pilot'")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--splits", type=int, default=5, help="complementarity: folds per repeat")
    ap.add_argument("--repeats", type=int, default=10, help="complementarity: repeats")
    ap.add_argument("--mode", default="per_band")
    ap.add_argument("--stages", nargs="+", default=["benchmark", "complementarity"],
                    choices=["benchmark", "complementarity"])
    ap.add_argument("--no-ztf", action="store_true",
                    help="skip the ZTF re-measurement (then the cross-sample comparison "
                         "falls back to the published figures only)")
    ap.add_argument("--no-minirocket", action="store_true")
    args = ap.parse_args()

    out = ROOT / "outputs"
    gaia = load_gaia(args.tag)
    payload: dict = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "seed": SEED,
        "folds": args.folds,
        "stages": args.stages,
        "mode": args.mode,
        "gaia": {"bands": list(GAIA_BANDS), "sample": describe(gaia, "gaia")},
        "published_ztf_reference": published_ztf_reference(),
    }
    _dump(payload, out)

    ztf = None if args.no_ztf else load_ztf()
    if ztf is not None:
        payload["ztf"] = {"bands": list(ZTF_BANDS), "sample": describe(ztf, "ztf")}

    if "benchmark" in args.stages:
        payload["gaia"]["representations"] = run_variant_grid(
            gaia, GAIA_BANDS, args.folds, args.mode, "gaia")
        payload["gaia"]["headline"] = derive_headline(
            payload["gaia"]["representations"], args.mode)
        _dump(payload, out)

        # MiniRocket is deliberately last in this stage. It is the most expensive single
        # evaluation here (about ten thousand features) and no conclusion depends on it, so
        # the P1 and P2 verdicts are computed and written before it runs.
        if not args.no_minirocket:
            add_minirocket(gaia, GAIA_BANDS, args.folds,
                           payload["gaia"]["representations"], "gaia")
            _dump(payload, out)

        if ztf is not None:
            payload["ztf"]["representations"] = ztf_reference_grid(
                ztf, args.folds, args.mode)
            payload["ztf"]["headline"] = derive_headline(
                payload["ztf"]["representations"], args.mode)
        # The P1/P2 verdicts are complete at this point, so they are written before the long
        # complementarity stage starts rather than after it, and cannot be lost to it.
        payload["comparison"] = build_comparison(payload)
        _dump(payload, out)
        write_table(payload, out)
        report(payload)

    if "complementarity" in args.stages:
        try:
            payload["gaia"]["complementarity"] = run_complementarity(
                gaia, GAIA_BANDS, args.splits, args.repeats, "gaia")
        except Exception as exc:                                # noqa: BLE001
            # Recorded, not swallowed: the stage is expensive and the benchmark verdicts
            # above stand on their own, but a silent omission would look like a clean run.
            import traceback
            traceback.print_exc()
            payload["gaia"]["complementarity_error"] = f"{type(exc).__name__}: {exc}"
        _dump(payload, out)

    payload["comparison"] = build_comparison(payload)
    _dump(payload, out)
    table = write_table(payload, out)
    if not table.empty:
        print("\n" + table.to_string(index=False))
    report(payload)
    return 0


def build_comparison(payload: dict) -> dict:
    """State the two predictions and whether each held, on the numbers just measured."""
    g = payload.get("gaia", {})
    z = payload.get("ztf", {})
    pub = payload.get("published_ztf_reference", {})
    comp: dict = {"note": (
        "Every entry is a within-sample margin measured on identical folds. Absolute "
        "balanced accuracies are not compared across samples, because the Gaia curves are "
        "denser and roughly an order of magnitude longer-baselined than the ZTF curves.")}

    gh, zh = g.get("headline"), z.get("headline")
    if gh:
        entry = {
            "gaia": gh["gap_baseline_minus_best_signature"],
            "gaia_best_signature": gh["best_signature_key"],
            "ztf_published": pub.get("benchmark", {}).get(
                "gap_baseline_minus_best_signature"),
        }
        if "baseline_without_absolute_photometry" in gh:
            entry["gaia_gap_with_baseline_stripped_of_absolute_photometry"] = (
                gh["baseline_without_absolute_photometry"]["gap_against_best_signature"])
        if zh and "baseline_without_absolute_photometry" in zh:
            entry["ztf_gap_with_baseline_stripped_of_absolute_photometry"] = (
                zh["baseline_without_absolute_photometry"]["gap_against_best_signature"])
        if zh:
            entry["ztf_remeasured"] = zh["gap_baseline_minus_best_signature"]
            entry["ztf_best_signature"] = zh["best_signature_key"]
            entry["gaia_minus_ztf"] = round(
                gh["gap_baseline_minus_best_signature"]
                - zh["gap_baseline_minus_best_signature"], 4)
            entry["P1_gap_larger_on_gaia"] = bool(
                gh["gap_baseline_minus_best_signature"]
                > zh["gap_baseline_minus_best_signature"])
        comp["gap_baseline_minus_best_signature"] = entry

        t = {"gaia": gh["time_channel"]["value_of_keeping_duration"],
             "gaia_all_pairs": gh["time_channel"]["all_pairs"],
             "gaia_all_pairs_mean": gh["time_channel"]["all_pairs_mean"],
             "gaia_all_pairs_max": gh["time_channel"]["all_pairs_max"]}
        if zh:
            t["ztf_remeasured"] = zh["time_channel"]["value_of_keeping_duration"]
            t["ztf_all_pairs"] = zh["time_channel"]["all_pairs"]
            t["ztf_all_pairs_mean"] = zh["time_channel"]["all_pairs_mean"]
            t["gaia_minus_ztf"] = round(t["gaia"] - t["ztf_remeasured"], 4)
            t["P2_time_channel_worth_more_on_gaia"] = bool(t["gaia"] > t["ztf_remeasured"])
            t["P2_on_pair_means"] = bool(
                t["gaia_all_pairs_mean"] > t["ztf_all_pairs_mean"])
        t["ztf_published_readme"] = 0.069
        t["what_is_compared"] = (
            "norm='raw_time' minus norm='unit_time' at per-band depth 4 with no "
            "augmentation; unit_time rescales each object's time axis to [0, 1] and so "
            "discards duration, raw_time keeps it in days")
        t["span_coefficient_of_variation"] = {
            "gaia": g.get("sample", {}).get("span_coefficient_of_variation"),
            "ztf": z.get("sample", {}).get("span_coefficient_of_variation"),
            "why_it_matters": (
                "unit_time can only destroy information if the observed baseline differs "
                "between objects; on a survey whose window is set by the mission rather "
                "than by the star, it does not")}
        comp["value_of_explicit_time_channel"] = t

        if "value_of_lead_lag" in gh:
            ll = {"gaia": gh["value_of_lead_lag"]["value"]}
            if zh and "value_of_lead_lag" in zh:
                ll["ztf_remeasured"] = zh["value_of_lead_lag"]["value"]
                ll["gaia_minus_ztf"] = round(ll["gaia"] - ll["ztf_remeasured"], 4)
            ll["what_is_compared"] = (
                "use_lead_lag=True minus use_lead_lag=False at per-band raw_time depth 4")
            comp["value_of_lead_lag"] = ll

    gc = g.get("complementarity")
    if gc:
        comp["complementarity"] = {
            "gaia_signature_gain":
                gc["summary_vs_summary_plus_signature"]["mean_difference"],
            "gaia_signature_gain_ci95":
                gc["summary_vs_summary_plus_signature"]["ci95"],
            "gaia_noise_cost": gc["summary_vs_summary_plus_noise"]["mean_difference"],
            "gaia_paired_gap_baseline_minus_signature":
                gc["paired_gap_baseline_minus_signature"],
            "ztf_signature_gain": pub.get("complementarity", {}).get("signature_gain"),
            "ztf_noise_cost": pub.get("complementarity", {}).get("noise_cost"),
            "ztf_paired_gap_baseline_minus_signature":
                pub.get("complementarity", {}).get(
                    "paired_gap_baseline_minus_signature"),
            "gaia_complementary": gc["verdict"]["complementary"],
        }
        zp = pub.get("complementarity", {}).get("paired_gap_baseline_minus_signature")
        if zp is not None:
            comp["complementarity"]["P1_paired_gap_larger_on_gaia"] = bool(
                gc["paired_gap_baseline_minus_signature"] > zp)
    return comp


def report(payload: dict) -> None:
    c = payload.get("comparison", {})
    print("\n" + "=" * 78)
    print("COUNTER-TEST VERDICT")
    print("=" * 78)
    gap = c.get("gap_baseline_minus_best_signature")
    if gap:
        print("P1  baseline minus best signature:")
        print(f"      gaia {gap['gaia']:+.4f}   ztf(re-measured) "
              f"{gap.get('ztf_remeasured', float('nan')):+.4f}   "
              f"ztf(published) {gap.get('ztf_published')}")
        print(f"      P1 (gap larger on gaia): {gap.get('P1_gap_larger_on_gaia')}")
    t = c.get("value_of_explicit_time_channel")
    if t:
        print("P2  value of keeping duration (raw_time minus unit_time), depth 4, no aug:")
        print(f"      gaia {t['gaia']:+.4f}   ztf(re-measured) "
              f"{t.get('ztf_remeasured', float('nan')):+.4f}   "
              f"ztf(README) +{t['ztf_published_readme']:.3f}")
        print(f"      across all depth/augmentation pairs, mean: "
              f"gaia {t.get('gaia_all_pairs_mean')}, ztf {t.get('ztf_all_pairs_mean')}")
        print(f"      gaia pairs: {t.get('gaia_all_pairs')}")
        print(f"      ztf  pairs: {t.get('ztf_all_pairs')}")
        print(f"      P2 (worth more on gaia): "
              f"{t.get('P2_time_channel_worth_more_on_gaia')}  "
              f"(on pair means: {t.get('P2_on_pair_means')})")
    ll = c.get("value_of_lead_lag")
    if ll:
        print(f"    value of the lead-lag transform: gaia {ll['gaia']:+.4f}   "
              f"ztf(re-measured) {ll.get('ztf_remeasured', float('nan')):+.4f}")
    k = c.get("complementarity")
    if k:
        print("complementarity, paired folds with matched-noise control:")
        print(f"      gaia signature {k['gaia_signature_gain']:+.4f}, "
              f"noise {k['gaia_noise_cost']:+.4f}")
        print(f"      ztf  signature {k['ztf_signature_gain']:+.4f}, "
              f"noise {k['ztf_noise_cost']:+.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
