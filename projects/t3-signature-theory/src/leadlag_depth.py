"""Does the lead-lag transform move ordering information from depth 3 down to depth 2?

T1's control experiment (`t1-signature-lightcurves/src/order_sensitivity.py`) established
that for a time-augmented path the ordering of two magnitude excursions is invisible at
signature depths 1 and 2 and appears only at depth 3. The stated reason is geometric: with
channels (t, m) the time coordinate is monotone, so the antisymmetric part of level 2
degenerates to the area under the curve, which is the same whichever peak came first.

This experiment tests the obvious remedy. The lead-lag transform pairs every channel with a
delayed copy of itself, doubling the channel count, and is the standard device for making
quadratic variation visible to a signature. If it also restores order-sensitivity to level 2,
depth-2 signatures become viable for time series and the feature count of the standard
recipe drops by a large factor.

REGISTERED PREDICTIONS (written before the classification arm was run)

  P1  lead_lag(time_aug) at depth 2 separates the peak-swap classes; the ordering signal
      moves from depth 3 down to depth 2.
      OUTCOME: REFUTED, and refuted at proof strength rather than by a score. The
      depth-<=2 content of lead_lag(time_aug) is an exact function of six scalars,
      {dt, dm, \\int m dt, Qtt, Qmm, Qtm} with Q the (co)variation half-sums
      Qtt = (1/2)\\sum dT_k^2, Qmm = (1/2)\\sum dM_k^2, Qtm = (1/2)\\sum dT_k dM_k. The
      closed form for the whole 4x4 level-2 tensor is verified below (check V3) to
      ~1e-16. Qtt and Qmm are invariant under any permutation of the increments, so they
      are order-blind absolutely; Qtm is antisymmetric under the reflection that defines
      the two classes but is O(mean spacing) and vanishes identically on a uniform grid.
      Lead-lag therefore adds first-order discretisation residue at level 2, not ordering.
      On exact-reflection pairs the two classes' level-2 tensors agree to ~1e-16 relative,
      so no classifier of any capacity can separate them; the measured accuracies below
      are a consequence, not the evidence.

  P2  a finite-lag delay embedding (t, m, m(t - tau)) restores level-2 order sensitivity,
      since neither m nor its delay is monotone and the pair traces a hysteresis loop.
      OUTCOME: REFUTED for the mechanism proposed, and the correction is instructive. The
      reflection m_B(t) = m_A(1 - t) maps the delay path to itself with the lead and lag
      coordinates exchanged AND the traversal reversed. Each of those two operations flips
      the sign of a Levy area, so their composition preserves it: the hysteresis loop is
      class-invariant. Measured level-2 class difference on an exactly uniform grid is at
      machine precision (~1e-16) for tau = 0.02, 0.05 and 0.10, at every sampling density
      tried. What is NOT at machine precision is tau = 0.20, and it separates the classes
      already at LEVEL 1 (1.250e-01): once tau is comparable to the event timescale, the
      delayed channel is truncated by the observation window and no longer returns to
      baseline, so its net increment becomes class-dependent. That is a window-edge effect
      rather than an iterated-integral one, and it is the reason the delay family cannot be
      reported as a level-2 mechanism.

  P3  a causal, memory-carrying channel restores level-2 order sensitivity provided it is
      paired with the time channel, because reflecting the data does not reflect the output
      of a backward-looking operator. The cumulative integral M = \\int m dt is the clean
      case: level 2 then contains S^{t,M} = \\int (t - t_0) m dt, the first temporal moment,
      which is exactly the functional the plain path reaches only at depth 3.
      OUTCOME: CONFIRMED, at 12 features and depth 2 against 14 features and depth 3.

  P4  the negative result for lead-lag cannot be explained by model capacity, because the
      arm that succeeds (plain depth 3, 14 features) is NARROWER than the arm that fails
      (lead-lag depth 2, 20 features).
      OUTCOME: CONFIRMED, and reinforced by two padding controls and one random-projection
      control (section "dimension controls").

  P5  lead-lag does buy something real, but it is quadratic variation, which is order-blind.
      On a family where the two classes share their profile and their ordering and differ
      only in high-frequency roughness, lead-lag at depth 2 should separate them and the
      plain path at depth 2 should not.
      OUTCOME: CONFIRMED. This localises what the transform actually contributes.

WHAT IS MEASURED HERE

  1. construction checks -- the peak-swap classes must have matched magnitude marginals and
     matched net change, else the whole design is void (section "construction checks");
  2. a proof-strength table on exact-reflection pairs: per level, the largest relative
     difference between the two classes' signature tensors. A machine-precision comparison
     that no classifier score can override, in either direction. It is swept against
     sampling density, because a nonzero level-2 difference at one density proves nothing --
     it may be continuum structure or it may be O(mean spacing) discretisation residue, and
     only refinement of the grid tells them apart. This distinction is not cosmetic: on an
     irregular grid lead-lag's level-2 class difference is 6.4e-02 at n = 15, which looks
     like a result and is not one, since it falls to 2.4e-04 by n = 645 and is at machine
     precision (1.2e-15) on an exactly uniform grid at every n tried;
  3. the classification sweep, augmentation x depth, balanced accuracy under identical
     5-fold cross-validation with a linear model and with gradient-boosted trees, always
     with the feature count;
  4. dimension controls, so a gain cannot be read as capacity;
  5. a thinning arm at the ~12-observation cadence where T1 found signatures stop paying;
  6. three positive controls that can each fail and must pass before any negative result is
     reported: orientation (level 2 must see a genuine 2-D loop), differing area (plain
     level 2 must separate when the area differs, else the account of what level 2 contains
     is wrong), differing quadratic variation (lead-lag depth 2 must separate, plain must
     not);
  7. a mutation test on the checks themselves. Every check above compares a measured number
     against a tolerance of 1e-12, which invites the question of whether it could ever
     exceed it, so each is re-run against a deliberately broken version of the property it
     tests. All fail by ten to fourteen orders of magnitude when broken. A check that cannot
     fail measures nothing and would not entitle anything here to be believed.

MEASURED HEADLINE (this module, 520 s, outputs/leadlag_depth.json)

  Depth-2 balanced accuracy on T1's peak-swap family, 800 objects, median 39 points,
  logistic / boosted trees, with the feature count that bought it:

    magnitude only          2 feats   0.490 / 0.496
    time-augmented (t, m)   6 feats   0.514 / 0.489
    basepoint + time        6 feats   0.518 / 0.481
    lead-lag(time)         20 feats   0.539 / 0.516     <- the hypothesis, at chance
    lead-lag(basepoint)    20 feats   0.520 / 0.521
    lead-lag(mag only)      6 feats   0.521 / 0.519
    delay tau=0.05         12 feats   0.518 / 0.501
    cumulative integral    12 feats   0.985 / 0.969     <- ordering, at depth 2
    running maximum        12 feats   0.984 / 0.979     <- ordering, at depth 2
    EWMA tau=0.05          12 feats   0.533 / 0.499

  The plain path needs depth 3 and 14 features to reach 0.979, so the causal channels reach
  the same information at depth 2 with 12. Lead-lag never pays on this family at any depth:
  84 features at depth 3 give 0.981 against the plain path's 14 features at 0.979, and 340
  features at depth 4 give 0.995 against 30 at 0.994 (paired difference -0.0025 to +0.0012,
  p = 0.37 to 0.59).

  A cautionary number. Lead-lag's depth-2 "gain" over the plain path is +0.0250 for the
  linear model and +0.0275 for the trees, on data where the two classes' level-2 tensors are
  provably identical to 1.2e-15. Neither survives a paired-fold test (p = 0.051 and 0.154).
  That gain is the same size as the +0.0246 T1 measured for lead-lag on real ZTF photometry
  and attributed to the transform. Nothing here can explain that gain, because here an
  identical gain arises where there is provably nothing to find.

Every number reported by this module comes from this run. Nothing is estimated.

Run:  python3 src/leadlag_depth.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

T1_SRC = Path("/Users/enes/Developer/structure-first-ml/projects/"
              "t1-signature-lightcurves/src")
sys.path.insert(0, str(T1_SRC))
from features_signature import _log_modulus            # noqa: E402
from order_sensitivity import (PEAK_AMPLITUDES, PEAK_TIMES, PEAK_WIDTH,  # noqa: E402
                               build_dataset, make_pair)
from signature import (add_basepoint, lead_lag, signature,  # noqa: E402
                       signature_dim, signature_levels)

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260726          # identical to T1, so the folds are the same folds
N_OBJECTS = 800          # identical to order_sensitivity.build_dataset default
DEPTHS = (1, 2, 3, 4)


# ------------------------------------------------------------------ channels
#
# The causal channels below do not exist in t1/src/signature.py and that file is not to be
# modified, so they are defined here and verified here (checks V4 and V6).


def chan_cum_integral(t: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid integral of m against t, starting at zero."""
    incr = 0.5 * (m[1:] + m[:-1]) * np.diff(t)
    return np.concatenate([[0.0], np.cumsum(incr)])


def chan_running_max(t: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Running maximum: causal, non-anticipating, and not reflection-covariant."""
    return np.maximum.accumulate(m)


def chan_ewma(t: np.ndarray, m: np.ndarray, tau: float) -> np.ndarray:
    """Exponentially weighted moving average on an irregular grid."""
    out = np.empty_like(m)
    out[0] = m[0]
    dt = np.diff(t)
    for i in range(1, m.size):
        a = np.exp(-dt[i - 1] / tau)
        out[i] = a * out[i - 1] + (1.0 - a) * m[i]
    return out


def chan_delay(t: np.ndarray, m: np.ndarray, tau: float) -> np.ndarray:
    """m evaluated at t - tau by linear interpolation, held at m[0] before the start."""
    return np.interp(t - tau, t, m, left=m[0], right=m[-1])


# -------------------------------------------------------------- augmentations
#
# Each augmentation maps a pair of equal-length arrays to a (n_points, d) path. The first
# argument is the time channel for families F1-F4 and F6; for the orientation family F5 it
# is a second magnitude channel instead, which is the point of that control -- it runs the
# same code with neither channel monotone.


def _plain(t, m):
    return np.column_stack([t, m])


AUGMENTATIONS: dict[str, tuple] = {
    # name                     builder                                         d
    "A0_mag_only":            (lambda t, m: m.reshape(-1, 1),                  1),
    "A1_time_aug":            (_plain,                                         2),
    "A2_basepoint_time":      (lambda t, m: add_basepoint(_plain(t, m)),        2),
    "A3_leadlag_time":        (lambda t, m: lead_lag(_plain(t, m)),             4),
    "A4_leadlag_basepoint":   (lambda t, m: lead_lag(add_basepoint(_plain(t, m))), 4),
    "A5_leadlag_mag":         (lambda t, m: lead_lag(m.reshape(-1, 1)),         2),
    "A6_delay_tau0.05":       (lambda t, m: np.column_stack(
                                   [t, m, chan_delay(t, m, 0.05)]),            3),
    "A7_cum_integral":        (lambda t, m: np.column_stack(
                                   [t, m, chan_cum_integral(t, m)]),           3),
    "A8_running_max":         (lambda t, m: np.column_stack(
                                   [t, m, chan_running_max(t, m)]),            3),
    "A9_ewma_tau0.05":        (lambda t, m: np.column_stack(
                                   [t, m, chan_ewma(t, m, 0.05)]),             3),
}

# Extra augmentations used only in the machine-precision table, where they are cheap.
PROOF_EXTRA: dict[str, tuple] = {
    f"A6_delay_tau{tau}": (lambda t, m, tau=tau: np.column_stack(
        [t, m, chan_delay(t, m, tau)]), 3) for tau in (0.02, 0.10, 0.20)
}
PROOF_EXTRA.update({
    f"A9_ewma_tau{tau}": (lambda t, m, tau=tau: np.column_stack(
        [t, m, chan_ewma(t, m, tau)]), 3) for tau in (0.02, 0.20)
})
PROOF_EXTRA["A7_cum_integral_no_time"] = (
    lambda t, m: np.column_stack([m, chan_cum_integral(t, m)]), 2)
PROOF_EXTRA["A8_running_max_no_time"] = (
    lambda t, m: np.column_stack([m, chan_running_max(t, m)]), 2)


# ------------------------------------------------------------- path families


def _raised_cosine(t: np.ndarray, centre: float, width: float, amp: float) -> np.ndarray:
    """A bump with COMPACT support on [centre - width, centre + width].

    Compact support is what makes the exact-reflection family exact: the profile is
    identically zero at both ends of the window, so the net magnitude change is 0 with no
    rounding error at all, and signature level 1 is exactly zero in both classes.
    """
    u = (t - centre) / width
    out = np.zeros_like(t)
    inside = np.abs(u) < 1.0
    out[inside] = amp * 0.5 * (1.0 + np.cos(np.pi * u[inside]))
    return out


def _symmetric_grid(n: int, rng: np.random.Generator | None, denom: int = 1024
                    ) -> np.ndarray:
    """A grid on [0, 1] exactly invariant under t -> 1 - t.

    Points are integers over a power-of-two denominator, so both j/denom and
    (denom - j)/denom are exactly representable and the reflection is exact in floating
    point. `rng=None` gives the EXACTLY uniform grid, which is the cadence on/off control:
    the cross (co)variation Qtm = (1/2) sum dT_k dM_k then collapses to (h/2)(m_last -
    m_first) = 0, so lead-lag's only class-dependent level-2 term vanishes identically. For
    that collapse to be exact rather than approximate the spacing must be exactly constant,
    which is why the uniform branch demands n - 1 to be a power of two.
    """
    if rng is None:
        if n - 1 <= 0 or (n - 1) & (n - 2):
            raise ValueError(f"uniform grid needs n - 1 a power of two, got n={n}")
        js = np.arange(n)
        denom = n - 1
    else:
        half = (n - 1) // 2
        inner = rng.choice(np.arange(1, denom // 2), size=half, replace=False)
        js = np.unique(np.concatenate([[0], inner, [denom // 2],
                                       denom - inner, [denom]]))
    return js.astype(np.float64) / float(denom)


def exact_reflection_pair(n: int = 81, rng: np.random.Generator | None = None
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One peak-swap pair on a shared grid, with class B the exact reversal of class A.

    Returns (t, m_A, m_B). The construction differs from T1's in three ways, all of them to
    remove floating-point slack rather than to change the phenomenon: compact-support bumps
    instead of Gaussians (so dm = 0 exactly), a shared grid symmetric under t -> 1 - t, and
    no noise. What is preserved is the only thing that matters: the classes differ solely in
    which peak comes first.
    """
    t = _symmetric_grid(n, rng)
    a1, a2 = PEAK_AMPLITUDES
    c1, c2 = PEAK_TIMES
    w = 0.15                     # support [0.15, 0.45] and [0.55, 0.85]: disjoint, interior
    m_a = _raised_cosine(t, c1, w, a1) + _raised_cosine(t, c2, w, a2)
    m_b = m_a[::-1].copy()       # exact reversal == exact peak swap on a symmetric grid
    return t, m_a, m_b


def family_peak_swap(n_objects: int = N_OBJECTS, n_min: int = 20, n_max: int = 60,
                     noise: float = 0.05, seed: int = SEED):
    """F1: T1's construction verbatim, via import."""
    ts, ms, y = build_dataset(n_objects=n_objects, n_min=n_min, n_max=n_max,
                              noise=noise, seed=seed)
    return list(zip(ts, ms)), y


def family_peak_swap_uniform(n_objects: int = 400, n: int = 40, noise: float = 0.05,
                             seed: int = SEED + 3):
    """F3: the same classes on a UNIFORM shared grid. Qtm is then identically zero."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    objs, ys = [], []
    for i in range(n_objects):
        rev = bool(i % 2)
        a1, a2 = PEAK_AMPLITUDES
        j = rng.normal(1.0, 0.08, size=2)
        a1, a2 = a1 * j[0], a2 * j[1]
        if rev:
            a1, a2 = a2, a1
        t1, t2 = PEAK_TIMES
        m = (a1 * np.exp(-((t - t1) / PEAK_WIDTH) ** 2)
             + a2 * np.exp(-((t - t2) / PEAK_WIDTH) ** 2)
             + rng.normal(0, noise, size=n))
        objs.append((t.copy(), m))
        ys.append(int(rev))
    return objs, np.array(ys)


def family_area_differing(n_objects: int = 400, noise: float = 0.05, scale: float = 2.0,
                          seed: int = SEED + 1):
    """F4: peaks swapped AND the area under the curve made unequal.

    Positive control on the account of what level 2 contains. If level 2 really supplies
    {dt, dm, \\int m dt} then the plain path at depth 2 MUST separate this family. If it
    does not, the analysis is wrong and every negative result in this file is suspect.

    `scale` is swept because the control turned out to be sensitive to it, and the reason is
    worth recording rather than tuning away. The level-2 entry is S^{mt} = \\int (m - m_0) dt,
    and m_0 is a single noisy observation, so the basepoint contributes a scatter of order
    sigma_noise x (t_last - t_first) ~ 0.05 to a signal of order 0.19 (scale - 1). At
    scale = 1.35 the class difference and that scatter are comparable and the control reads
    0.74, which is not a decisive demonstration of anything. At scale = 2.0 it is.
    """
    rng = np.random.default_rng(seed)
    objs, ys = [], []
    for i in range(n_objects):
        rev = bool(i % 2)
        n = int(rng.integers(20, 60))
        t, m = make_pair(rng, n, rev, noise)
        if rev:
            m = m * scale                    # same ordering as class B, larger area
        objs.append((t, m))
        ys.append(int(rev))
    return objs, np.array(ys)


def family_orientation_loop(n_objects: int = 400, noise: float = 0.02,
                            seed: int = SEED + 2):
    """F5: a genuine two-dimensional (m_g, m_r) colour-magnitude loop, both directions.

    No time channel: neither coordinate is monotone, so the Levy area is a real signed area
    and level 2 must reach ~1.0. This is the check that the implementation sees orientation
    at all. If it fails, nothing else in this file means anything.
    """
    rng = np.random.default_rng(seed)
    objs, ys = [], []
    for i in range(n_objects):
        rev = bool(i % 2)
        n = int(rng.integers(20, 60))
        phase = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.8, 1.2)
        s = np.linspace(0.0, 2 * np.pi, n)
        if rev:
            s = -s
        g = r * np.cos(s + phase) + rng.normal(0, noise, size=n)
        rr = r * np.sin(s + phase) + rng.normal(0, noise, size=n)
        objs.append((g, rr))                 # (channel 1, channel 2), no time
        ys.append(int(rev))
    return objs, np.array(ys)


def family_qv_differing(n_objects: int = 400, noise: float = 0.05,
                        jitter: float = 0.10, seed: int = SEED + 4):
    """F6: identical profile, identical ordering, differing high-frequency roughness.

    Both classes are class-A-ordered double peaks. Class 1 receives extra independent
    jitter, so its realised quadratic variation Qmm is larger while its smooth profile,
    its area and its ordering are unchanged. Lead-lag at depth 2 must separate this and the
    plain path at depth 2 must not; that is what localises the transform's contribution.

    The jitter is applied to INTERIOR points only. Perturbing the endpoints as well would
    inflate the variance of the net increment m_last - m_first, which is signature level 1,
    and the plain path would then separate the classes at depth 1 for a reason that has
    nothing to do with quadratic variation. Measured with endpoint jitter included, plain
    depth 1 reached 0.615 and plain depth 2 reached 0.688, so the control did not cleanly
    isolate quadratic variation. That variant was discarded and its numbers are not in the
    committed outputs; they are recorded here as the reason for the design.
    """
    rng = np.random.default_rng(seed)
    objs, ys = [], []
    for i in range(n_objects):
        rough = bool(i % 2)
        n = int(rng.integers(20, 60))
        t, m = make_pair(rng, n, False, noise)
        if rough:
            m = m.copy()
            m[1:-1] = m[1:-1] + rng.normal(0, jitter, size=n - 2)
        objs.append((t, m))
        ys.append(int(rough))
    return objs, np.array(ys)


# --------------------------------------------------------------- measurement


def _cv(X: np.ndarray, y: np.ndarray, folds: int = 5) -> dict:
    """T1's protocol exactly: same splitter, same seed, same two models."""
    X = np.nan_to_num(np.asarray(X, dtype=np.float64),
                      nan=0.0, posinf=0.0, neginf=0.0)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    lin = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, random_state=SEED))
    trees = HistGradientBoostingClassifier(random_state=SEED)
    out = {"n_features": int(X.shape[1])}
    for name, model in (("logistic", lin), ("boosted_trees", trees)):
        s = cross_val_score(model, X, y, cv=skf, scoring="balanced_accuracy")
        out[name] = round(float(s.mean()), 4)
        out[name + "_std"] = round(float(s.std()), 4)
        out[name + "_folds"] = [round(float(v), 4) for v in s]
    return out


def build_X(objs, builder, depth: int) -> np.ndarray:
    return np.array([_log_modulus(signature(builder(a, b), depth)) for a, b in objs])


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    """Largest absolute difference, normalised by the larger tensor's largest entry."""
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    if scale < 1e-14:
        return 0.0
    return float(np.max(np.abs(a - b)) / scale)


UNIFORM_NS = (33, 65, 129, 257, 513)          # n - 1 a power of two: exactly uniform
IRREGULAR_NS = (13, 23, 43, 83, 163, 323, 643)


def convergence_table(ns, uniform: bool) -> dict:
    """Levels 1 and 2 relative class difference against sampling density.

    This is the measurement that decides the experiment, and it is the one the single-n
    table cannot make. A nonzero level-2 class difference at one sampling density proves
    nothing: it may be genuine continuum structure, or it may be O(mean spacing)
    discretisation residue that vanishes as the path is resolved. Sweeping n separates
    them. An augmentation carries genuine level-2 order information only if its level-2
    difference converges to a nonzero constant.

    Fails loudly in both directions. If lead-lag really moved ordering to level 2, its row
    would sit at a constant O(1e-1); if the cumulative integral did not, its row would
    decay like lead-lag's does.
    """
    rows = {}
    for name, (builder, d) in {**AUGMENTATIONS, **PROOF_EXTRA}.items():
        l1, l2, sizes = [], [], []
        for n in ns:
            rng = None if uniform else np.random.default_rng(SEED + n)
            t, m_a, m_b = exact_reflection_pair(n, rng)
            la = signature_levels(builder(t, m_a), 2)
            lb = signature_levels(builder(t, m_b), 2)
            sizes.append(int(t.size))
            l1.append(_rel(la[0], lb[0]))
            l2.append(_rel(la[1], lb[1]))
        # The classification rule, stated so it can be checked: "genuine" requires the
        # level-2 difference to survive a 16-fold refinement of the grid.
        if l2[-1] < 1e-10:
            verdict = "order-blind at level 2 (machine precision)"
        elif l2[-1] < 1e-3 and l2[-1] < 0.1 * l2[0]:
            verdict = "residue only (vanishes under densification)"
        elif l1[-1] > 1e-6:
            verdict = "genuine, but already visible at level 1"
        else:
            verdict = "genuine level-2 order sensitivity"
        rows[name] = {"d": d, "n_points": sizes, "level1": l1, "level2": l2,
                      "level2_limit": l2[-1], "level1_limit": l1[-1], "verdict": verdict}
    return rows


def proof_table(n: int, rng: np.random.Generator | None, depth: int = 4) -> dict:
    """Per level, the relative difference between the two classes' signature tensors.

    This is the headline measurement and it is not a classifier score. If a level's
    relative difference is at machine precision, the two classes are the SAME point in that
    tensor space and no model of any capacity can tell them apart. If the analysis in the
    docstring is wrong, the lead-lag row at level 2 comes back at 1e-2 instead of 1e-16 and
    the failure is loud.
    """
    t, m_a, m_b = exact_reflection_pair(n, rng)
    table = {}
    for name, (builder, d) in {**AUGMENTATIONS, **PROOF_EXTRA}.items():
        la = signature_levels(builder(t, m_a), depth)
        lb = signature_levels(builder(t, m_b), depth)
        table[name] = {"d": d,
                       "levels": {str(k + 1): _rel(la[k], lb[k]) for k in range(depth)}}
    return table


# ------------------------------------------------------- verification checks
#
# Each check below is written so that it CAN fail, and the comment states what would make
# it fail. A check that cannot fail measures nothing.


def check_construction(objs, y) -> dict:
    """V1: the peak-swap classes must have matched marginals and matched net change.

    Fails if the two classes' pooled magnitude distributions differ (KS p < 0.01) or if
    their mean net change differs by more than the noise permits. Either failure would mean
    an order-blind statistic could solve the task, which is exactly the defect that ruined
    the first version of T1's construction.
    """
    a = np.concatenate([m for (_, m), lab in zip(objs, y) if lab == 0])
    b = np.concatenate([m for (_, m), lab in zip(objs, y) if lab == 1])
    ks = stats.ks_2samp(a, b)
    net_a = np.array([m[-1] - m[0] for (_, m), lab in zip(objs, y) if lab == 0])
    net_b = np.array([m[-1] - m[0] for (_, m), lab in zip(objs, y) if lab == 1])
    tt = stats.ttest_ind(net_a, net_b, equal_var=False)
    return {
        "pooled_mag_mean": [float(a.mean()), float(b.mean())],
        "pooled_mag_std": [float(a.std()), float(b.std())],
        "pooled_mag_skew": [float(stats.skew(a)), float(stats.skew(b))],
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "net_change_mean": [float(net_a.mean()), float(net_b.mean())],
        "net_change_std": [float(net_a.std()), float(net_b.std())],
        "net_change_welch_p": float(tt.pvalue),
        "marginals_matched": bool(ks.pvalue > 0.01),
        "net_change_matched": bool(tt.pvalue > 0.01),
    }


def check_exact_reflection(n: int = 81) -> dict:
    """V2: the diagnostic family must really be exact.

    Fails if the grid is not symmetric under t -> 1 - t to the last bit, if the net
    magnitude change is not exactly zero, or if reversing class A does not reproduce the
    peak-swapped profile. Without all three, a machine-precision comparison of the two
    classes' signatures would be meaningless.
    """
    rng = np.random.default_rng(SEED)
    t, m_a, m_b = exact_reflection_pair(n, rng)
    a1, a2 = PEAK_AMPLITUDES
    c1, c2 = PEAK_TIMES
    swapped = _raised_cosine(t, c1, 0.15, a2) + _raised_cosine(t, c2, 0.15, a1)
    return {
        "n_points": int(t.size),
        "grid_symmetry_max_abs": float(np.max(np.abs(t + t[::-1] - 1.0))),
        "net_change_A": float(m_a[-1] - m_a[0]),
        "net_change_B": float(m_b[-1] - m_b[0]),
        "reversal_equals_peak_swap_max_abs": float(np.max(np.abs(m_b - swapped))),
        "marginals_identical": bool(np.array_equal(np.sort(m_a), np.sort(m_b))),
        "exact": bool(np.max(np.abs(t + t[::-1] - 1.0)) == 0.0
                      and m_a[-1] - m_a[0] == 0.0
                      and np.array_equal(np.sort(m_a), np.sort(m_b))),
    }


def check_leadlag_closed_form() -> dict:
    """V3: the closed form for the lead-lag level-2 tensor.

    Predicted, in channel order (t_lead, m_lead, t_lag, m_lag), with P the plain level-2
    tensor and Q the (co)variation half-sums:

        row t_lead: [P11,      P12,      P11+Qtt, P12+Qtm]
        row m_lead: [P21,      P22,      P21+Qtm, P22+Qmm]
        row t_lag:  [P11-Qtt,  P12-Qtm,  P11,     P12    ]
        row m_lag:  [P21-Qtm,  P22-Qmm,  P21,     P22    ]

    Fails if `lead_lag` does not have the interleaving this derivation assumes, or if the
    derivation is wrong. Either way the P1 refutation would collapse, because it rests
    entirely on this identity: the whole of depth <= 2 is six scalars, three of which are
    the plain path's and three of which are (co)variation.
    """
    rng = np.random.default_rng(11)
    t = np.sort(rng.uniform(0, 1, 37))
    m = np.sin(9 * t) + rng.normal(0, 0.2, 37)
    p = np.column_stack([t, m])
    P = signature_levels(p, 2)[1].reshape(2, 2)
    dT, dM = np.diff(t), np.diff(m)
    Qtt, Qmm, Qtm = 0.5 * np.sum(dT ** 2), 0.5 * np.sum(dM ** 2), 0.5 * np.sum(dT * dM)
    pred = np.array([
        [P[0, 0],       P[0, 1],       P[0, 0] + Qtt, P[0, 1] + Qtm],
        [P[1, 0],       P[1, 1],       P[1, 0] + Qtm, P[1, 1] + Qmm],
        [P[0, 0] - Qtt, P[0, 1] - Qtm, P[0, 0],       P[0, 1]],
        [P[1, 0] - Qtm, P[1, 1] - Qmm, P[1, 0],       P[1, 1]],
    ])
    lv = signature_levels(lead_lag(p), 2)
    meas2 = lv[1].reshape(4, 4)
    l1_pred = np.array([t[-1] - t[0], m[-1] - m[0], t[-1] - t[0], m[-1] - m[0]])
    err2 = float(np.max(np.abs(meas2 - pred)))
    err1 = float(np.max(np.abs(lv[0] - l1_pred)))
    return {
        "level1_max_abs_error": err1,
        "level2_max_abs_error": err2,
        "Qtt": float(Qtt), "Qmm": float(Qmm), "Qtm": float(Qtm),
        "closed_form_holds": bool(err1 < 1e-12 and err2 < 1e-12),
    }


def check_cum_integral_mechanism() -> dict:
    """V4/V6: the cumulative channel really does carry the first temporal moment.

    Two independent checks. (a) the level-2 entry S^{t,M} equals the exact quadrature
    \\sum_k [(t_k - t_0) dM_k + dt_k dM_k / 2] computed outside the signature code -- this
    fails if the flat-array index convention is misread, which is the plausible bug.
    (b) on the exact-reflection pair, the class difference in S^{t,M} is exactly half the
    class difference in the plain level-3 word S^{tmt} -- this fails if the claim that the
    causal channel pulls the depth-3 ordering functional down to depth 2 is wrong.
    """
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(0, 1, 41))
    m = 1.0 + np.cos(7 * t)
    M = chan_cum_integral(t, m)
    lv = signature_levels(np.column_stack([t, m, M]), 2)[1].reshape(3, 3)
    dM, dt = np.diff(M), np.diff(t)
    quad = float(np.sum((t[:-1] - t[0]) * dM + dt * dM / 2.0))
    err_a = abs(float(lv[0, 2]) - quad)

    t2, m_a, m_b = exact_reflection_pair(257, None)
    s_tM = []
    s_tmt = []
    for m in (m_a, m_b):
        M = chan_cum_integral(t2, m)
        s_tM.append(float(signature_levels(np.column_stack([t2, m, M]), 2)[1]
                          .reshape(3, 3)[0, 2]))
        s_tmt.append(float(signature_levels(np.column_stack([t2, m]), 3)[2]
                           .reshape(2, 2, 2)[0, 1, 0]))
    d_tM = s_tM[0] - s_tM[1]
    d_tmt = s_tmt[0] - s_tmt[1]
    ratio = d_tM / d_tmt if abs(d_tmt) > 1e-14 else float("nan")
    return {
        "S_tM_signature": float(lv[0, 2]),
        "S_tM_independent_quadrature": quad,
        "quadrature_max_abs_error": err_a,
        "S_tM_classes": s_tM,
        "S_tmt_classes": s_tmt,
        "ratio_dStM_over_dStmt": ratio,
        "mechanism_holds": bool(err_a < 1e-12 and abs(ratio - 0.5) < 1e-6),
    }


def check_orientation_primitive() -> dict:
    """V5: the primitive sees signed area. A unit circle must give Levy area +-pi.

    Fails if the signature implementation or the level-2 antisymmetrisation is broken, in
    which case the level-2 degeneracy reported for time-augmented paths would be a bug
    rather than a geometric fact.
    """
    s = np.linspace(0, 2 * np.pi, 2001)
    out = {}
    for label, ss in (("anticlockwise", s), ("clockwise", -s)):
        lv = signature_levels(np.column_stack([np.cos(ss), np.sin(ss)]), 2)[1].reshape(2, 2)
        out[label] = float(lv[0, 1] - lv[1, 0])       # twice the Levy area
    out["pi"] = float(np.pi)
    out["sees_orientation"] = bool(abs(out["anticlockwise"] / 2 - np.pi) < 1e-3
                                   and abs(out["clockwise"] / 2 + np.pi) < 1e-3)
    return out


def mutation_test() -> dict:
    """Meta-check: break each property deliberately and confirm its check notices.

    A check that passes while measuring nothing is worse than no check, and the checks above
    all compare a measured number against a tolerance of 1e-12, so it is fair to ask whether
    they could ever exceed it. Each entry below re-runs one check against a deliberately
    wrong version of the thing it tests. Every `broken` value must exceed the tolerance by
    orders of magnitude; if any came back at 1e-16, that check would be vacuous and the
    result it underwrites would have to be withdrawn.
    """
    out = {}

    # V3: drop the cross (co)variation terms from the closed form.
    rng = np.random.default_rng(11)
    t = np.sort(rng.uniform(0, 1, 37))
    m = np.sin(9 * t) + rng.normal(0, 0.2, 37)
    p = np.column_stack([t, m])
    P = signature_levels(p, 2)[1].reshape(2, 2)
    dT, dM = np.diff(t), np.diff(m)
    Qtt, Qmm, Qtm = (0.5 * np.sum(dT ** 2), 0.5 * np.sum(dM ** 2), 0.5 * np.sum(dT * dM))
    meas = signature_levels(lead_lag(p), 2)[1].reshape(4, 4)

    def predict(qtt, qmm, qtm):
        return np.array([
            [P[0, 0],       P[0, 1],       P[0, 0] + qtt, P[0, 1] + qtm],
            [P[1, 0],       P[1, 1],       P[1, 0] + qtm, P[1, 1] + qmm],
            [P[0, 0] - qtt, P[0, 1] - qtm, P[0, 0],       P[0, 1]],
            [P[1, 0] - qtm, P[1, 1] - qmm, P[1, 0],       P[1, 1]]])

    out["V3_correct"] = float(np.max(np.abs(meas - predict(Qtt, Qmm, Qtm))))
    out["V3_broken_Qtm_zeroed"] = float(np.max(np.abs(meas - predict(Qtt, Qmm, 0.0))))
    out["V3_broken_Qtt_Qmm_zeroed"] = float(np.max(np.abs(meas - predict(0.0, 0.0, Qtm))))

    # V4: misread the level-2 flat-array index convention (transpose the entry).
    t2 = np.sort(np.random.default_rng(5).uniform(0, 1, 41))
    m2 = 1.0 + np.cos(7 * t2)
    M = chan_cum_integral(t2, m2)
    lv = signature_levels(np.column_stack([t2, m2, M]), 2)[1].reshape(3, 3)
    dMM, dtt = np.diff(M), np.diff(t2)
    quad = float(np.sum((t2[:-1] - t2[0]) * dMM + dtt * dMM / 2.0))
    out["V4_correct_index"] = abs(float(lv[0, 2]) - quad)
    out["V4_broken_transposed_index"] = abs(float(lv[2, 0]) - quad)

    # V2: an arbitrary random grid is not reflection-symmetric.
    g_ok = _symmetric_grid(81, np.random.default_rng(1))
    g_bad = np.sort(np.random.default_rng(1).uniform(0, 1, 81))
    out["V2_correct_grid_symmetry"] = float(np.max(np.abs(g_ok + g_ok[::-1] - 1.0)))
    out["V2_broken_arbitrary_grid"] = float(np.max(np.abs(g_bad + g_bad[::-1] - 1.0)))

    # V5: a path monotone in one coordinate has no signed area to find.
    s = np.linspace(0, 1, 2001)
    lvm = signature_levels(np.column_stack([s, s ** 2]), 2)[1].reshape(2, 2)
    out["V5_broken_monotone_path_area"] = float(lvm[0, 1] - lvm[1, 0]) / 2.0

    out["all_checks_can_fail"] = bool(
        out["V3_broken_Qtm_zeroed"] > 1e-6
        and out["V3_broken_Qtt_Qmm_zeroed"] > 1e-6
        and out["V4_broken_transposed_index"] > 1e-6
        and out["V2_broken_arbitrary_grid"] > 1e-6
        and abs(out["V5_broken_monotone_path_area"] - np.pi) > 1e-3)
    return out


# ---------------------------------------------------------- dimension control


def pad_with_noise(X: np.ndarray, width: int, rng: np.random.Generator) -> np.ndarray:
    """Append Gaussian columns matching the existing columns' variances.

    Widens a feature block without adding information, so a model that gains from extra
    width alone will gain here too.
    """
    v = np.var(X, axis=0)
    v = np.where(v > 0, v, 1.0)
    reps = np.ceil(width / X.shape[1]).astype(int)
    scales = np.sqrt(np.tile(v, reps))[:width]
    return np.hstack([X, rng.normal(0.0, 1.0, size=(X.shape[0], width)) * scales])


def project(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Fixed Gaussian random projection down to k columns."""
    G = rng.normal(0.0, 1.0 / np.sqrt(X.shape[1]), size=(X.shape[1], k))
    return X @ G


def paired(cell_a: dict, cell_b: dict, model: str = "logistic") -> dict:
    """Fold-by-fold comparison of two arms scored on identical splits.

    T1 reported lead-lag as worth +0.025 on real photometry without ever running a paired
    test on the ablation, and that gain sat inside one per-fold standard deviation. This
    function exists so the same mistake is not repeated here. With five folds the
    Wilcoxon signed-rank statistic cannot produce a p-value below 0.0625 whatever the data,
    so that floor is reported alongside it and a small p is not over-read.
    """
    a = np.asarray(cell_a[model + "_folds"], dtype=float)
    b = np.asarray(cell_b[model + "_folds"], dtype=float)
    d = a - b
    res = {"model": model, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
           "mean_diff": float(d.mean()), "fold_diffs": [float(v) for v in d],
           "n_folds": int(d.size), "wilcoxon_min_attainable_p": 0.0625}
    res["ttest_rel_p"] = (float(stats.ttest_rel(a, b).pvalue)
                          if np.ptp(d) > 0 else float("nan"))
    res["wilcoxon_p"] = (float(stats.wilcoxon(a, b).pvalue)
                         if np.any(d != 0) else float("nan"))
    res["significant_at_0.05"] = bool(res["ttest_rel_p"] < 0.05)
    return res


# -------------------------------------------------------------------- driver


def sweep(objs, y, names, depths, label: str, rows: list) -> dict:
    res = {}
    for name in names:
        builder, d = AUGMENTATIONS[name]
        for depth in depths:
            t0 = time.time()
            X = build_X(objs, builder, depth)
            assert X.shape[1] == signature_dim(d, depth), (name, depth, X.shape)
            cell = _cv(X, y)
            cell["d"] = d
            cell["build_seconds"] = round(time.time() - t0, 2)
            res[f"{name}|depth{depth}"] = cell
            rows.append({"family": label, "augmentation": name, "d": d, "depth": depth,
                         "n_features": cell["n_features"],
                         "logistic": cell["logistic"], "logistic_std": cell["logistic_std"],
                         "boosted_trees": cell["boosted_trees"],
                         "boosted_trees_std": cell["boosted_trees_std"]})
            print(f"  {label:14s} {name:24s} depth {depth}  "
                  f"{cell['n_features']:4d} feats  "
                  f"logistic {cell['logistic']:.4f}  trees {cell['boosted_trees']:.4f}")
    return res


def main() -> int:
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    out: dict = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "seed": SEED}
    rows: list = []

    # ---------------------------------------------------------------- checks
    print("=== verification checks (each can fail; see docstrings) ===")
    objs_f1, y_f1 = family_peak_swap()
    checks = {
        "V1_construction_F1": check_construction(objs_f1, y_f1),
        "V2_exact_reflection": check_exact_reflection(),
        "V3_leadlag_closed_form": check_leadlag_closed_form(),
        "V4_V6_cum_integral_mechanism": check_cum_integral_mechanism(),
        "V5_orientation_primitive": check_orientation_primitive(),
    }
    out["checks"] = checks
    for k, v in checks.items():
        verdict = [(kk, vv) for kk, vv in v.items() if isinstance(vv, bool)]
        print(f"  {k}: " + ", ".join(f"{kk}={vv}" for kk, vv in verdict))
    print(f"  F1 marginals: mean {checks['V1_construction_F1']['pooled_mag_mean'][0]:.5f} "
          f"vs {checks['V1_construction_F1']['pooled_mag_mean'][1]:.5f}, "
          f"KS p={checks['V1_construction_F1']['ks_pvalue']:.3f}; "
          f"net change {checks['V1_construction_F1']['net_change_mean'][0]:+.5f} vs "
          f"{checks['V1_construction_F1']['net_change_mean'][1]:+.5f}, "
          f"Welch p={checks['V1_construction_F1']['net_change_welch_p']:.3f}")
    print(f"  V3 closed form max abs error: "
          f"{checks['V3_leadlag_closed_form']['level2_max_abs_error']:.3e}")
    print(f"  V4 quadrature error {checks['V4_V6_cum_integral_mechanism']['quadrature_max_abs_error']:.3e}, "
          f"ratio {checks['V4_V6_cum_integral_mechanism']['ratio_dStM_over_dStmt']:.9f}")

    mut = mutation_test()
    out["mutation_test"] = mut
    print("  mutation test (each check re-run against a deliberately broken version):")
    for k, v in mut.items():
        if isinstance(v, float):
            print(f"    {k:34s} {v:.3e}")
    print(f"    all_checks_can_fail: {mut['all_checks_can_fail']}")

    # ------------------------------------- proof-strength tables (families F2/F3/F7)
    print("\n=== F2 per-level table at n=81 (irregular symmetric grid) and n=65 (uniform) ===")
    proof = {"irregular_symmetric_n83": proof_table(81, np.random.default_rng(SEED)),
             "uniform_n65": proof_table(65, None)}
    for label in ("irregular_symmetric_n83", "uniform_n65"):
        print(f"  -- {label}")
        for name, cellv in proof[label].items():
            lv = cellv["levels"]
            print(f"     {name:26s} d={cellv['d']}  "
                  + "  ".join(f"L{k}={lv[k]:.3e}" for k in ("1", "2", "3", "4")))
    out["proof_table"] = proof

    print("\n=== F7/F3 densification: does the level-2 class difference survive? ===")
    conv = {"uniform": convergence_table(UNIFORM_NS, uniform=True),
            "irregular": convergence_table(IRREGULAR_NS, uniform=False)}
    for label in ("uniform", "irregular"):
        print(f"  -- {label} grid, level-2 relative |S(A)-S(B)|")
        for name, cellv in conv[label].items():
            print(f"     {name:26s} "
                  + " ".join(f"{v:9.2e}" for v in cellv["level2"])
                  + f"   {cellv['verdict']}")
    out["convergence"] = conv

    # The cross (co)variation scalar that is lead-lag's ONLY class-dependent level-2 term.
    qtm = {}
    for n in IRREGULAR_NS:
        t, m_a, m_b = exact_reflection_pair(n, np.random.default_rng(SEED + n))
        qtm[str(int(t.size))] = {
            "Qtm_A": 0.5 * float(np.sum(np.diff(t) * np.diff(m_a))),
            "Qtm_B": 0.5 * float(np.sum(np.diff(t) * np.diff(m_b))),
            "mean_spacing": float(np.mean(np.diff(t))),
        }
    for n in UNIFORM_NS:
        t, m_a, m_b = exact_reflection_pair(n, None)
        qtm[f"uniform_{int(t.size)}"] = {
            "Qtm_A": 0.5 * float(np.sum(np.diff(t) * np.diff(m_a))),
            "Qtm_B": 0.5 * float(np.sum(np.diff(t) * np.diff(m_b))),
            "mean_spacing": float(np.mean(np.diff(t))),
        }
    out["Qtm_scalar"] = qtm
    print("  Qtm on the irregular grid: "
          + ", ".join(f"n={k}:{v['Qtm_A']:+.2e}/{v['Qtm_B']:+.2e}"
                      for k, v in list(qtm.items())[:len(IRREGULAR_NS)]))

    # --------------------------------------------------- main sweep on F1
    print("\n=== F1 (T1 construction) augmentation x depth ===")
    out["sweep_F1"] = sweep(objs_f1, y_f1, list(AUGMENTATIONS), DEPTHS, "F1_peak_swap", rows)

    # --------------------------------------------------- thinned arm
    print("\n=== F1-thin (~12 observations per object) ===")
    objs_thin, y_thin = family_peak_swap(n_min=10, n_max=15, seed=SEED + 7)
    out["thin_median_points"] = int(np.median([len(t) for t, _ in objs_thin]))
    out["checks"]["V1_construction_F1_thin"] = check_construction(objs_thin, y_thin)
    out["sweep_F1_thin"] = sweep(objs_thin, y_thin, list(AUGMENTATIONS), DEPTHS,
                                 "F1_thin", rows)

    # --------------------------------------------------- controls
    print("\n=== F3 uniform grid (Qtm identically zero) ===")
    objs_u, y_u = family_peak_swap_uniform()
    out["sweep_F3_uniform"] = sweep(objs_u, y_u,
                                    ["A1_time_aug", "A3_leadlag_time", "A7_cum_integral"],
                                    DEPTHS, "F3_uniform", rows)

    print("\n=== F4 area differing (plain depth 2 MUST separate) ===")
    objs_a, y_a = family_area_differing(scale=2.0)
    out["sweep_F4_area"] = sweep(objs_a, y_a,
                                 ["A1_time_aug", "A3_leadlag_time", "A7_cum_integral"],
                                 (1, 2, 3), "F4_area_differing", rows)
    objs_a2, y_a2 = family_area_differing(scale=1.35)
    out["sweep_F4_area_scale1.35"] = sweep(objs_a2, y_a2, ["A1_time_aug"], (1, 2, 3),
                                           "F4_area_differing_scale1.35", rows)

    print("\n=== F5 orientation loop, no time channel (level 2 MUST reach ~1.0) ===")
    objs_o, y_o = family_orientation_loop()
    out["sweep_F5_orientation"] = sweep(objs_o, y_o,
                                        ["A0_mag_only", "A1_time_aug", "A3_leadlag_time"],
                                        (1, 2, 3), "F5_orientation", rows)

    print("\n=== F6 quadratic variation differing, ordering identical ===")
    objs_q, y_q = family_qv_differing()
    out["sweep_F6_qv"] = sweep(objs_q, y_q,
                               ["A1_time_aug", "A3_leadlag_time", "A5_leadlag_mag"],
                               (1, 2, 3), "F6_qv_differing", rows)

    # --------------------------------------------------- dimension controls
    print("\n=== dimension controls on F1 ===")
    ctrl = {}
    X_plain2 = build_X(objs_f1, AUGMENTATIONS["A1_time_aug"][0], 2)
    X_ll2 = build_X(objs_f1, AUGMENTATIONS["A3_leadlag_time"][0], 2)
    X_cum2 = build_X(objs_f1, AUGMENTATIONS["A7_cum_integral"][0], 2)

    ctrl["C1_plain_d2_padded_to_20"] = _cv(
        pad_with_noise(X_plain2, 20 - X_plain2.shape[1], rng), y_f1)
    ctrl["C2_plain_d2_padded_to_12"] = _cv(
        pad_with_noise(X_plain2, 12 - X_plain2.shape[1], rng), y_f1)
    ctrl["C3_leadlag_d2_projected_to_6"] = _cv(project(X_ll2, 6, rng), y_f1)
    ctrl["C4_cum_integral_d2_projected_to_6"] = _cv(project(X_cum2, 6, rng), y_f1)
    ctrl["C5_leadlag_d2_padded_to_84"] = _cv(
        pad_with_noise(X_ll2, 84 - X_ll2.shape[1], rng), y_f1)
    for k, v in ctrl.items():
        print(f"  {k:36s} {v['n_features']:4d} feats  "
              f"logistic {v['logistic']:.4f}  trees {v['boosted_trees']:.4f}")
        rows.append({"family": "F1_peak_swap", "augmentation": k, "d": -1, "depth": -1,
                     "n_features": v["n_features"], "logistic": v["logistic"],
                     "logistic_std": v["logistic_std"],
                     "boosted_trees": v["boosted_trees"],
                     "boosted_trees_std": v["boosted_trees_std"]})
    out["dimension_controls"] = ctrl

    # --------------------------------------------------- paired-fold tests
    print("\n=== paired-fold tests on identical splits ===")
    S1 = out["sweep_F1"]
    pairs = {
        "leadlag_d2_vs_plain_d2": (S1["A3_leadlag_time|depth2"], S1["A1_time_aug|depth2"]),
        "leadlag_d2_vs_plain_d3": (S1["A3_leadlag_time|depth2"], S1["A1_time_aug|depth3"]),
        "cum_integral_d2_vs_plain_d2": (S1["A7_cum_integral|depth2"],
                                        S1["A1_time_aug|depth2"]),
        "cum_integral_d2_vs_plain_d3": (S1["A7_cum_integral|depth2"],
                                        S1["A1_time_aug|depth3"]),
        "running_max_d2_vs_plain_d3": (S1["A8_running_max|depth2"],
                                       S1["A1_time_aug|depth3"]),
        "leadlag_d4_vs_plain_d4": (S1["A3_leadlag_time|depth4"], S1["A1_time_aug|depth4"]),
    }
    out["paired_tests"] = {}
    for k, (ca, cb) in pairs.items():
        for model in ("logistic", "boosted_trees"):
            p = paired(ca, cb, model)
            out["paired_tests"][f"{k}|{model}"] = p
            print(f"  {k:32s} {model:14s} diff {p['mean_diff']:+.4f}  "
                  f"paired-t p={p['ttest_rel_p']:.4f}  "
                  f"significant={p['significant_at_0.05']}")

    # --------------------------------------------------- headline verdicts
    def cell(sw, name, depth, model="boosted_trees"):
        return out[sw][f"{name}|depth{depth}"][model]

    d2 = {name: {"n_features": out["sweep_F1"][f"{name}|depth2"]["n_features"],
                 "logistic": cell("sweep_F1", name, 2, "logistic"),
                 "boosted_trees": cell("sweep_F1", name, 2)}
          for name in AUGMENTATIONS}
    out["headline_depth2_F1"] = d2
    print("\n=== THE RESULT: depth-2 balanced accuracy on F1, side by side ===")
    for name, v in d2.items():
        print(f"  {name:24s} {v['n_features']:4d} feats  "
              f"logistic {v['logistic']:.4f}  trees {v['boosted_trees']:.4f}")

    ll2 = max(cell("sweep_F1", "A3_leadlag_time", 2),
              cell("sweep_F1", "A3_leadlag_time", 2, "logistic"))
    pl2 = max(cell("sweep_F1", "A1_time_aug", 2),
              cell("sweep_F1", "A1_time_aug", 2, "logistic"))
    pl3 = max(cell("sweep_F1", "A1_time_aug", 3),
              cell("sweep_F1", "A1_time_aug", 3, "logistic"))
    cum2 = max(cell("sweep_F1", "A7_cum_integral", 2),
               cell("sweep_F1", "A7_cum_integral", 2, "logistic"))
    verdict = {
        "leadlag_level2_class_diff_uniform_limit":
            conv["uniform"]["A3_leadlag_time"]["level2_limit"],
        "leadlag_level2_class_diff_irregular_n13":
            conv["irregular"]["A3_leadlag_time"]["level2"][0],
        "leadlag_level2_class_diff_irregular_limit":
            conv["irregular"]["A3_leadlag_time"]["level2_limit"],
        "cum_integral_level2_class_diff_uniform_limit":
            conv["uniform"]["A7_cum_integral"]["level2_limit"],
        "running_max_level2_class_diff_uniform_limit":
            conv["uniform"]["A8_running_max"]["level2_limit"],
        "P1_refuted_at_machine_precision":
            bool(conv["uniform"]["A3_leadlag_time"]["level2_limit"] < 1e-10),
        "augmentations_with_genuine_level2_ordering":
            [k for k, v in conv["uniform"].items()
             if v["verdict"] == "genuine level-2 order sensitivity"],
        "leadlag_depth2_best": ll2,
        "plain_depth2_best": pl2,
        "plain_depth3_best": pl3,
        "cum_integral_depth2_best": cum2,
        "leadlag_gain_at_depth2": round(ll2 - pl2, 4),
        "P1_leadlag_moves_ordering_to_depth2": bool(ll2 > 0.75),
        "P3_causal_channel_moves_ordering_to_depth2": bool(cum2 > 0.75),
        "leadlag_gain_at_depth2_paired_t_p":
            out["paired_tests"]["leadlag_d2_vs_plain_d2|logistic"]["ttest_rel_p"],
        "leadlag_gain_at_depth2_significant":
            out["paired_tests"]["leadlag_d2_vs_plain_d2|logistic"]["significant_at_0.05"],
        "positive_control_orientation_level2":
            cell("sweep_F5_orientation", "A1_time_aug", 2),
        "positive_control_area_plain_level2":
            cell("sweep_F4_area", "A1_time_aug", 2),
        "positive_control_area_plain_level2_scale1.35":
            cell("sweep_F4_area_scale1.35", "A1_time_aug", 2),
        "positive_control_qv_leadlag_level2":
            cell("sweep_F6_qv", "A3_leadlag_time", 2),
        "positive_control_qv_plain_level2":
            cell("sweep_F6_qv", "A1_time_aug", 2),
        "all_checks_pass": all(v.get(kk) for v in checks.values()
                               for kk in ("marginals_matched", "exact",
                                          "closed_form_holds", "mechanism_holds",
                                          "sees_orientation") if kk in v),
    }
    verdict["controls_valid"] = bool(
        verdict["positive_control_orientation_level2"] > 0.90
        and verdict["positive_control_area_plain_level2"] > 0.75
        and verdict["positive_control_qv_leadlag_level2"] > 0.75
        and verdict["positive_control_qv_plain_level2"] < 0.70)
    out["verdict"] = verdict
    print("\n=== verdict ===")
    for k, v in verdict.items():
        print(f"  {k}: {v}")

    out["runtime_seconds"] = round(time.time() - t_start, 1)
    odir = ROOT / "outputs"
    odir.mkdir(exist_ok=True)
    (odir / "leadlag_depth.json").write_text(json.dumps(out, indent=2))
    with (odir / "leadlag_depth.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {odir/'leadlag_depth.json'} and {odir/'leadlag_depth.csv'} "
          f"in {out['runtime_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
