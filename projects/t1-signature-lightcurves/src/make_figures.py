"""Figures for the T1 report.

Two conventions are deliberate and worth stating.

Light curves are drawn as points with error bars and **no connecting line**. A line between
observations is an interpolation, and drawing one would assert continuity the data does not
contain -- which is the very assumption this track exists to avoid. The visual sparseness
is the point.

Colour is assigned to representations in a fixed order and never cycled. The four-slot
palette passes the lightness-band, chroma, colour-vision-deficiency separation,
normal-vision and contrast checks against the report surface.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
ASSETS = ROOT / "site" / "assets"

# Fixed assignment: representation -> colour. Never cycled, never reordered by rank.
COLOURS = {
    "summary": "#0072B2",
    "signature": "#D55E00",
    "minirocket": "#009E73",
    "combined": "#6A4C93",
}
BAND_COLOUR = {"g": "#0072B2", "r": "#D55E00"}
INK, MUTED, RULE = "#161616", "#5b5b5b", "#d7d7d2"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.edgecolor": RULE,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "grid.color": RULE,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def _classify(name: str) -> str:
    if name.startswith("signature"):
        return "signature"
    if name.startswith("summary+"):
        return "combined"
    if name.startswith("summary"):
        return "summary"
    return "minirocket"


def _tidy(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ figure 1

def fig_lightcurves(df: pd.DataFrame) -> None:
    """Example light curves, one per class, showing the real sampling."""
    classes = df.groupby("label").oid.nunique().sort_values(ascending=False).index[:6]
    n = len(classes)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(9.5, 5.2), squeeze=False)
    rng = np.random.default_rng(7)

    for ax, cls in zip(axes.ravel(), classes):
        pool = df[df.label == cls]
        counts = pool.groupby("oid").size()
        # pick a median-length example rather than the best-looking one
        target = counts.iloc[np.argsort(np.abs(counts.values - counts.median()))[0]]
        oid = counts[counts == target].index[rng.integers(0, (counts == target).sum())]
        lc = pool[pool.oid == oid]
        t0 = lc.mjd.min()
        for band in ("g", "r"):
            sub = lc[lc.band == band]
            ax.errorbar(sub.mjd - t0, sub.mag, yerr=sub.magerr, fmt="o", ms=3.0,
                        lw=0, elinewidth=0.7, capsize=0, color=BAND_COLOUR[band],
                        label=band, alpha=0.9, zorder=3)
        ax.invert_yaxis()
        ax.set_title(f"{cls}  ({len(lc)} obs)", color=INK)
        _tidy(ax)

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("days from first detection")
    for ax in axes[:, 0]:
        ax.set_ylabel("magnitude")
    axes[0, 0].legend(loc="lower right", title=None)

    fig.suptitle("ZTF light curves as observed: points only, no interpolating line",
                 fontsize=10.5, color=INK, y=0.99)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig1_lightcurves.png")
    plt.close(fig)
    print("  fig1_lightcurves.png")


# ------------------------------------------------------------------ figure 2

def fig_sampling(df: pd.DataFrame) -> None:
    """The irregularity itself: gap distribution and observation counts."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))

    gaps = []
    for _, lc in df.groupby(["oid", "band"]):
        t = np.sort(lc.mjd.to_numpy())
        if len(t) > 1:
            gaps.append(np.diff(t))
    gaps = np.concatenate(gaps)
    gaps = gaps[gaps > 0]

    axes[0].hist(np.log10(gaps), bins=60, color=COLOURS["summary"], alpha=0.85, zorder=3)
    axes[0].set_xlabel("log$_{10}$ gap between consecutive observations (days)")
    axes[0].set_ylabel("count")
    med = np.median(gaps)
    axes[0].axvline(np.log10(med), color=INK, lw=1.0, ls="--", zorder=4)
    axes[0].annotate(f"median {med:.1f} d", xy=(np.log10(med), 0.92),
                     xycoords=("data", "axes fraction"), xytext=(6, 0),
                     textcoords="offset points", fontsize=8.5, color=INK)
    axes[0].set_title("Sampling gaps span four orders of magnitude", color=INK)
    _tidy(axes[0])

    counts = df.groupby("oid").size()
    axes[1].hist(counts, bins=50, color=COLOURS["signature"], alpha=0.85, zorder=3)
    axes[1].set_xlabel("detections per object")
    axes[1].set_ylabel("count")
    axes[1].axvline(counts.median(), color=INK, lw=1.0, ls="--", zorder=4)
    axes[1].annotate(f"median {int(counts.median())}", xy=(counts.median(), 0.92),
                     xycoords=("data", "axes fraction"), xytext=(6, 0),
                     textcoords="offset points", fontsize=8.5, color=INK)
    axes[1].set_title("Light-curve lengths are short and uneven", color=INK)
    _tidy(axes[1])

    fig.tight_layout()
    fig.savefig(ASSETS / "fig2_sampling.png")
    plt.close(fig)
    print("  fig2_sampling.png")


# ------------------------------------------------------------------ figure 3

def fig_benchmark(table: pd.DataFrame) -> None:
    """Head-to-head comparison of representations, best variant per family."""
    table = table.copy()
    table["family"] = [_classify(i) for i in table.index]
    best = table.sort_values("balanced_accuracy", ascending=False).groupby("family").head(1)
    best = best.sort_values("balanced_accuracy")

    fig, ax = plt.subplots(figsize=(7.6, 0.55 * len(best) + 1.6))
    y = np.arange(len(best))
    ax.barh(y, best.balanced_accuracy, height=0.55, zorder=3,
            color=[COLOURS[f] for f in best.family])
    for yi, (name, row) in zip(y, best.iterrows()):
        ax.annotate(f"{row.balanced_accuracy:.3f}",
                    xy=(row.balanced_accuracy, yi), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(best.index, fontsize=8.5)
    ax.set_xlabel("balanced accuracy (5-fold cross-validation, gradient-boosted trees)")
    ax.set_xlim(0, min(1.0, best.balanced_accuracy.max() * 1.22))
    _tidy(ax)
    ax.grid(True, axis="x", alpha=0.5, zorder=0)
    ax.grid(False, axis="y")
    ax.set_title("Best variant of each representation family", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(ASSETS / "fig3_benchmark.png")
    plt.close(fig)
    print("  fig3_benchmark.png")


# ------------------------------------------------------------------ figure 4

def fig_ablation(ab: pd.DataFrame) -> None:
    """The experiment: degradation under random and blocked thinning."""
    ab = ab.copy()
    ab["family"] = [_classify(r) for r in ab.representation]
    regimes = ["random", "blocked"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), sharey=True)

    for ax, regime in zip(axes, regimes):
        sub = ab[ab.regime == regime]
        for fam, grp in sub.groupby("family"):
            grp = grp.sort_values("fraction")
            ax.plot(grp.fraction, grp.balanced_accuracy, "-o", ms=4.5, lw=1.6,
                    color=COLOURS[fam], label=fam, zorder=3)
            ax.fill_between(grp.fraction,
                            grp.balanced_accuracy - grp.balanced_accuracy_std,
                            grp.balanced_accuracy + grp.balanced_accuracy_std,
                            color=COLOURS[fam], alpha=0.12, lw=0, zorder=2)
        ax.set_xlabel("fraction of observations retained")
        ax.set_title(f"{regime} thinning", color=INK)
        ax.invert_xaxis()
        _tidy(ax)

    axes[0].set_ylabel("balanced accuracy")
    axes[0].legend(loc="lower left")
    fig.suptitle("Same number of observations removed, different gap structure",
                 fontsize=10.5, color=INK, y=1.01)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig4_ablation.png")
    plt.close(fig)
    print("  fig4_ablation.png")


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    made = 0

    lc_path = ROOT / "data" / "ztf_bts_lightcurves.parquet"
    if lc_path.exists():
        df = pd.read_parquet(lc_path)
        print(f"light curves: {df.oid.nunique()} objects")
        fig_lightcurves(df)
        fig_sampling(df)
        made += 2
    else:
        print(f"skipping light-curve figures, missing {lc_path}", file=sys.stderr)

    bench = OUT / "benchmark_table.csv"
    if bench.exists():
        fig_benchmark(pd.read_csv(bench, index_col=0))
        made += 1
    else:
        print("skipping benchmark figure, run src/run_benchmark.py first", file=sys.stderr)

    abl = OUT / "ablation_sampling.csv"
    if abl.exists():
        fig_ablation(pd.read_csv(abl))
        made += 1
    else:
        print("skipping ablation figure, run src/ablate_sampling.py first", file=sys.stderr)

    print(f"{made} figures written to {ASSETS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
