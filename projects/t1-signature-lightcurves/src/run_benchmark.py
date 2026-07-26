"""Benchmark signature features against the incumbent baselines on ZTF BTS transients.

Every representation is evaluated on identical cross-validation folds with identical
classifiers, so that differences are attributable to the representation and nothing else.

Representations compared:

  summary      Hand-crafted variability features (src/features_baseline.py). Mostly
               order-blind: functions of the marginal magnitude distribution.
  signature-*  Truncated path signatures under the three multi-band path constructions
               and a range of truncation depths.
  minirocket   Random convolutional kernels on linearly interpolated, fixed-length light
               curves. MiniRocket needs a regular grid, so it is given one -- that
               interpolation step is exactly the cost the signature approach claims to
               avoid, and giving the baseline its best case is the only way to measure
               that cost honestly.
  summary+sig  The two feature sets concatenated, to test whether signatures add anything
               beyond what the incumbent already captures. This is the question that
               matters most: a representation that only matches the baseline is
               uninteresting, but one that is complementary to it is useful even if it
               loses head to head.

Run:  python3 src/run_benchmark.py [--depths 2 3 4] [--folds 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features_baseline import feature_frame  # noqa: E402
from features_signature import signature_feature_frame  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260726
GRID_LEN = 64  # points per band on the interpolated grid given to MiniRocket


# ----------------------------------------------------------------- MiniRocket input

def interpolated_panel(df: pd.DataFrame, bands: tuple[str, ...] = ("g", "r"),
                       length: int = GRID_LEN) -> tuple[np.ndarray, list, list]:
    """Linearly interpolate each object onto a fixed grid: (n_objects, n_bands, length).

    This is the standard preprocessing that convolutional and kernel methods require, and
    it is applied only to those baselines.
    """
    oids, labels, panels = [], [], []
    for oid, lc in df.groupby("oid", sort=True):
        chans = []
        for b in bands:
            sub = lc[lc.band == b].sort_values("mjd")
            t, m = sub.mjd.to_numpy(), sub.mag.to_numpy()
            if len(t) < 2:
                chans.append(np.zeros(length))
                continue
            grid = np.linspace(t.min(), t.max(), length)
            chans.append(np.interp(grid, t, m - np.median(m)))
        panels.append(np.vstack(chans))
        oids.append(oid)
        labels.append(lc.label.iloc[0])
    return np.stack(panels), oids, labels


def minirocket_features(df: pd.DataFrame) -> pd.DataFrame | None:
    try:
        from sktime.transformations.panel.rocket import MiniRocketMultivariate
    except ImportError:
        print("  sktime not available, skipping MiniRocket")
        return None
    panel, oids, labels = interpolated_panel(df)
    tr = MiniRocketMultivariate(random_state=SEED)
    feats = np.asarray(tr.fit_transform(panel))
    out = pd.DataFrame(feats, index=pd.Index(oids, name="oid"))
    out.columns = [f"mr{i}" for i in range(feats.shape[1])]
    out["label"] = labels
    return out


# ----------------------------------------------------------------- evaluation

def evaluate(X: pd.DataFrame, y: pd.Series, folds: int) -> dict:
    """Cross-validated balanced accuracy and macro F1 for two classifiers."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    Xv = np.nan_to_num(X.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    yv = y.to_numpy()

    models = {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced",
                               random_state=SEED)),
        "boosted_trees": HistGradientBoostingClassifier(
            random_state=SEED, class_weight="balanced"),
    }

    result = {"n_features": int(Xv.shape[1]), "n_objects": int(Xv.shape[0])}
    for name, model in models.items():
        bal, f1 = [], []
        for tr_idx, te_idx in skf.split(Xv, yv):
            model.fit(Xv[tr_idx], yv[tr_idx])
            pred = model.predict(Xv[te_idx])
            bal.append(balanced_accuracy_score(yv[te_idx], pred))
            f1.append(f1_score(yv[te_idx], pred, average="macro"))
        result[name] = {
            "balanced_accuracy": round(float(np.mean(bal)), 4),
            "balanced_accuracy_std": round(float(np.std(bal)), 4),
            "macro_f1": round(float(np.mean(f1)), 4),
            "macro_f1_std": round(float(np.std(f1)), 4),
        }
    return result


def _align(*frames: pd.DataFrame) -> tuple[list[pd.DataFrame], pd.Series]:
    """Restrict every feature frame to the objects present in all of them."""
    common = frames[0].index
    for f in frames[1:]:
        common = common.intersection(f.index)
    common = common.sort_values()
    label = frames[0].loc[common, "label"]
    return [f.loc[common].drop(columns=["label"]) for f in frames], label


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--modes", nargs="+", default=["per_band", "interleaved", "joint_ffill"])
    args = ap.parse_args()

    path = ROOT / "data" / "ztf_bts_lightcurves.parquet"
    if not path.exists():
        print(f"missing {path}; run src/fetch_ztf_bts.py first", file=sys.stderr)
        return 1
    df = pd.read_parquet(path)
    print(f"loaded {len(df)} detections, {df.oid.nunique()} objects, "
          f"{df.label.nunique()} classes")
    print(df.groupby("label").oid.nunique().sort_values(ascending=False).to_string())

    results: dict[str, dict] = {}

    print("\nsummary features ...")
    t0 = time.time()
    summary = feature_frame(df)
    print(f"  {summary.shape[1] - 1} features in {time.time() - t0:.1f} s")
    y = summary["label"]
    results["summary"] = evaluate(summary.drop(columns=["label"]), y, args.folds)
    _show("summary", results["summary"])

    best_sig_key, best_sig_frame, best_score = None, None, -1.0
    for mode in args.modes:
        for depth in args.depths:
            key = f"signature-{mode}-d{depth}"
            t0 = time.time()
            try:
                sig = signature_feature_frame(df, depth=depth, mode=mode)
            except Exception as exc:
                print(f"  {key} failed: {type(exc).__name__}: {exc}")
                continue
            (Xs,), ys = _align(sig)
            results[key] = evaluate(Xs, ys, args.folds)
            results[key]["build_seconds"] = round(time.time() - t0, 1)
            _show(key, results[key])
            score = results[key]["boosted_trees"]["balanced_accuracy"]
            if score > best_score:
                best_sig_key, best_sig_frame, best_score = key, sig, score

    print("\nMiniRocket (on interpolated grid) ...")
    mr = minirocket_features(df)
    if mr is not None:
        (Xm,), ym = _align(mr)
        results["minirocket"] = evaluate(Xm, ym, args.folds)
        _show("minirocket", results["minirocket"])

    if best_sig_frame is not None:
        print(f"\ncombined: summary + {best_sig_key} ...")
        (Xa, Xb), yc = _align(summary, best_sig_frame)
        combined = pd.concat([Xa, Xb], axis=1)
        results[f"summary+{best_sig_key}"] = evaluate(combined, yc, args.folds)
        results[f"summary+{best_sig_key}"]["note"] = (
            "concatenation of the hand-crafted features with the best signature variant")
        _show(f"summary+{best_sig_key}", results[f"summary+{best_sig_key}"])

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    payload = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "seed": SEED,
        "folds": args.folds,
        "n_objects": int(df.oid.nunique()),
        "class_counts": df.groupby("label").oid.nunique().to_dict(),
        "results": results,
    }
    (out / "benchmark_results.json").write_text(json.dumps(payload, indent=2))

    table = pd.DataFrame({
        k: {"balanced_accuracy": v["boosted_trees"]["balanced_accuracy"],
            "macro_f1": v["boosted_trees"]["macro_f1"],
            "logistic_balanced_accuracy": v["logistic"]["balanced_accuracy"],
            "n_features": v["n_features"]}
        for k, v in results.items()}).T
    table = table.sort_values("balanced_accuracy", ascending=False)
    table.to_csv(out / "benchmark_table.csv")
    print("\n" + table.to_string())
    return 0


def _show(name: str, r: dict) -> None:
    bt, lg = r["boosted_trees"], r["logistic"]
    print(f"  {name:34s} dim={r['n_features']:5d}  "
          f"trees bal-acc={bt['balanced_accuracy']:.4f}+-{bt['balanced_accuracy_std']:.4f}  "
          f"logistic bal-acc={lg['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
