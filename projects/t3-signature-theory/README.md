# T3 — Truncation depth, sampling irregularity, and where the information lives

**Status: skeleton, with one result already in hand.** The design below is unchanged, but
question 1 has been partly answered ahead of schedule by a control experiment run inside T1,
and the answer was not the one predicted. That result is recorded here because it belongs to
this track's subject matter, and because it changes what the remaining questions should ask.

## Overview

The theory track. T1 will produce a number — a classification score on real photometry using
truncated path signatures — and that number will not, on its own, explain anything. T3 exists to
supply the explanation using controlled synthetic data, where the ground truth is known by
construction and every nuisance factor can be varied one at a time.

## The question

The signature of a path $X:[0,T]\to\mathbb{R}^d$ is the infinite sequence of iterated integrals

$$S(X) = \left(1,\ \int_{0<t<T} dX_t,\ \iint_{0<s<t<T} dX_s\otimes dX_t,\ \dots\right),$$

whose $k$-th term lives in $(\mathbb{R}^d)^{\otimes k}$. The universality theorem states that linear
functionals of the signature approximate continuous functions on path space uniformly on compact
sets. In practice one keeps only terms up to depth $m$, giving $\sum_{k=0}^{m} d^k$ coefficients,
and the theorem's guarantee becomes an approximation whose quality depends on where the truncation
falls.

Three questions follow, none of which the theorem answers:

1. **Depth allocation.** For a given signal class, how does discriminative information distribute
   across depths? If depth 2 carries nearly everything, the universality argument is doing no work
   and the method is an expensive way to compute quadratic statistics. If depth 4 is where the
   separation appears, the argument has force.

2. **Sampling density and gap structure.** The signature is defined for a continuous path but
   computed from a discrete sample, conventionally by linear interpolation between observations. As
   sampling becomes sparser, the computed signature departs from that of the underlying path. The
   departure should depend not only on the number of observations but on the *structure* of the
   gaps — periodic gaps from diurnal and seasonal scheduling are not equivalent to random gaps of
   the same total measure. This is the crux: astronomical sampling gaps are highly structured, and
   claiming invariance to irregular sampling is only defensible if the structured case is tested.

3. **Noise and the invariance that is not there.** Reparameterisation invariance is exact for the
   underlying path. Heteroscedastic observational noise is not a reparameterisation, and the
   standard augmentations used to break the invariance where it is unwanted — adding a time
   coordinate, the lead-lag transform, basepoint augmentation — change the noise sensitivity in ways
   that are not documented for this regime.

## Result already obtained: ordering lives at depth 3, not depth 2

T1's control experiment (`projects/t1-signature-lightcurves/src/order_sensitivity.py`) built
two classes of synthetic double-peaked light curve differing *only* in which peak comes
first — identical marginal magnitude distributions, identical net change from first to last
observation, opposite ordering. The prediction, written down in advance, was that separation
would appear at depth 2, the Lévy area being the classical order-sensitive term.

| Representation | Balanced accuracy |
|---|---|
| Order-blind summary features | 0.512 |
| Signature depth 1 | 0.524 |
| Signature depth 2 | 0.489 |
| Signature depth 3 | **0.978** |
| Signature depth 4 | 0.990 |

Depths 1 and 2 sit at chance. The separation appears at depth 3.

**Why.** With channels $(t, m)$ the time coordinate is strictly increasing, so the
antisymmetric part of level 2 — the Lévy area — reduces to $\int m\,\mathrm{d}t$ once the
boundary terms vanish for a path returning to its starting magnitude. That is the area under
the light curve, which is identical whichever peak came first. **A path monotone in one
coordinate encloses no signed area that ordering can change.** Ordering first becomes visible
at level 3, in terms of the form $\iiint \mathrm{d}t\,\mathrm{d}m\,\mathrm{d}t$, which weight
magnitude excursions by when they occurred.

**Why this matters beyond the experiment.** Every time-augmented signature of a time series
has a monotone time channel — that is what time augmentation means. So this degeneracy is not
a quirk of one synthetic construction; it is a structural property of the standard way
signatures are applied to time series. The practical consequence is immediate: truncating at
depth 2 to economise on features discards precisely the ordering information the method is
supposed to supply, and any resulting null result would be misattributed to the data.

This also sharpens question 1 below. The question is no longer "how does information
distribute across depths" but the more specific: **which augmentations restore
order-sensitivity to level 2, and at what cost?** The lead-lag transform is the obvious
candidate, since it doubles the channels and breaks monotonicity — and T1 measured it as
worth +0.025 on real photometry without knowing why. Establishing whether that gain is the
same phenomenon is a concrete, bounded experiment.

## Planned method

1. **Synthetic path families with known separation.** Generate classes that differ in a controlled
   way: sinusoids differing in period, damped oscillators differing in damping ratio, stochastic
   paths differing in Hurst exponent, and shape-matched pairs that differ only in the *order* of
   their features — the case where signatures should win outright, since ordering information is
   exactly what iterated integrals encode and what any order-blind summary statistic discards.
2. **Sampling ablation.** Apply observation masks drawn from real survey cadences alongside matched
   random masks of identical density, isolating the effect of gap structure from that of gap
   fraction.
3. **Depth sweep.** Classify at truncation depths 1 through 5 with a fixed linear model, so that any
   change in performance is attributable to the representation rather than to model capacity.
4. **Per-depth attribution.** Measure how much of the achievable separation each depth contributes,
   including whether higher depths add anything once lower ones are present.
5. **Baseline parity.** Repeat the sweep with MiniRocket's random convolutional kernels at matched
   feature-count budgets. Both are cheap deterministic feature maps; the comparison isolates
   structured iterated integrals against random projections at equal cost.

## What would count as a result

A defensible outcome is a statement of the form: *for path families separated by feature ordering,
signature depth $m$ is necessary and sufficient, and the advantage over random-kernel baselines
survives up to gap fraction $f$ under structured masking but not beyond.* That is a claim with
conditions attached, and it can be checked by anyone who runs the code.

A null result — that depth 2 suffices everywhere tested, and that MiniRocket matches signatures at
equal feature budget — is equally publishable within this repository and would substantially change
how T1's results should be read. It is recorded either way.

## Results: lead-lag does not move ordering to depth 2, and a causal channel does

**The prediction was refuted.** `src/leadlag_depth.py`, run once end to end in 520 s
(`outputs/leadlag_depth.json`, `outputs/leadlag_depth.csv`). The lead-lag transform does not
restore order-sensitivity to signature level 2, and the failure is not marginal — it is
exact. The compensating positive result is that a different family of augmentations, causal
memory-carrying channels, does restore it, and reaches at depth 2 what the plain path needs
depth 3 to see.

### The result: depth-2 balanced accuracy, side by side

Family F1 is T1's construction imported unchanged from
`t1-signature-lightcurves/src/order_sensitivity.py` (800 objects, median 39 points, classes
differing only in which peak comes first). Identical `StratifiedKFold(5, random_state=20260726)`
splits throughout; signed log-modulus features; standardised logistic regression as the
primary model and `HistGradientBoostingClassifier` as the generous probe.

| Augmentation | $d$ | Features at depth 2 | Logistic | Boosted trees |
|---|---|---|---|---|
| magnitude only | 1 | 2 | 0.490 | 0.496 |
| time-augmented $(t,m)$ | 2 | 6 | 0.514 | 0.489 |
| basepoint + time | 2 | 6 | 0.518 | 0.481 |
| **lead-lag(time)** — the hypothesis | 4 | **20** | **0.539** | **0.516** |
| lead-lag(basepoint + time) | 4 | 20 | 0.520 | 0.521 |
| lead-lag(magnitude only) | 2 | 6 | 0.521 | 0.519 |
| delay $(t, m, m(t-0.05))$ | 3 | 12 | 0.518 | 0.501 |
| **cumulative integral $(t,m,M)$** | 3 | **12** | **0.985** | **0.969** |
| **running maximum** | 3 | **12** | **0.984** | **0.979** |
| EWMA, $\tau = 0.05$ | 3 | 12 | 0.533 | 0.499 |

Every lead-lag variant sits at chance. On the same data the plain path reaches 0.979 at
depth 3 (14 features) and 0.994 at depth 4 (30 features); lead-lag reaches 0.981 at depth 3
for 84 features and 0.995 at depth 4 for 340. **Lead-lag never pays on this family at any
depth.** The depth-4 comparison is a paired difference of $-0.0025$ to $+0.0012$
($p = 0.37$ to $0.59$) for eleven times the features.

### Why, at proof strength rather than by score

The whole of the lead-lag signature at depth $\le 2$ is an exact function of **six scalars**:
the three the plain path already has, $\{\Delta t,\ \Delta m,\ \int m\,\mathrm{d}t\}$, plus
the realised (co)variation half-sums

$$Q_{tt} = \tfrac12\textstyle\sum_k \Delta T_k^2,\qquad Q_{mm} = \tfrac12\sum_k \Delta M_k^2,\qquad Q_{tm} = \tfrac12\sum_k \Delta T_k\,\Delta M_k.$$

Writing $P$ for the plain level-2 tensor, the full $4\times 4$ lead-lag level-2 tensor in
channel order $(t_{\text{lead}}, m_{\text{lead}}, t_{\text{lag}}, m_{\text{lag}})$ is

$$\begin{pmatrix} P_{11} & P_{12} & P_{11}+Q_{tt} & P_{12}+Q_{tm}\\ P_{21} & P_{22} & P_{21}+Q_{tm} & P_{22}+Q_{mm}\\ P_{11}-Q_{tt} & P_{12}-Q_{tm} & P_{11} & P_{12}\\ P_{21}-Q_{tm} & P_{22}-Q_{mm} & P_{21} & P_{22} \end{pmatrix}$$

and level 1 is $(\Delta t, \Delta m, \Delta t, \Delta m)$, a duplicate carrying nothing.
This closed form was predicted first and then measured: maximum absolute discrepancy
**8.9e-16** over all sixteen entries (check `V3`). Equivalently, lead-lag replaces the
trapezoid area sum with the left-endpoint Riemann sum, and the difference is exactly $Q_{tm}$.

That settles it. $Q_{tt}$ and $Q_{mm}$ are invariant under any permutation of the
increments, so they are **order-blind absolutely**. $Q_{tm}$ does flip sign under the
reflection defining the two classes, but it equals $\tfrac{h}{2}(m_{\text{last}} -
m_{\text{first}})$ on a uniform grid and so vanishes identically, and off a uniform grid it
is $O(\text{mean spacing})$.

The measurement that establishes this is not a classifier score. On exact-reflection pairs
(compact-support raised-cosine bumps on a grid exactly invariant under $t \mapsto 1-t$, so
$\Delta m = 0$ **exactly** and class B is the bit-for-bit reversal of class A) the largest
relative difference between the two classes' level-2 tensors, against sampling density:

| Augmentation | $n=33$ | $n=65$ | $n=129$ | $n=257$ | $n=513$ | Reading |
|---|---|---|---|---|---|---|
| time-augmented $(t,m)$ | 2.8e-16 | 2.8e-16 | 2.6e-16 | 1.0e-16 | 1.2e-15 | order-blind at level 2 |
| **lead-lag(time)** | 3.2e-16 | 2.7e-16 | 4.4e-16 | 4.4e-16 | **1.2e-15** | **order-blind at level 2** |
| lead-lag(magnitude only) | 5.4e-16 | 7.0e-16 | 2.6e-15 | 1.3e-15 | 7.9e-15 | order-blind at level 2 |
| delay, $\tau = 0.05$ | 7.1e-16 | 3.9e-16 | 2.6e-16 | 6.4e-16 | 7.7e-16 | order-blind at level 2 |
| **cumulative integral** | 6.00e-02 | 6.00e-02 | 6.00e-02 | 6.00e-02 | **6.00e-02** | **genuine** |
| **running maximum** | 2.75e-01 | 2.76e-01 | 2.76e-01 | 2.76e-01 | **2.76e-01** | **genuine** |
| EWMA, $\tau = 0.05$ | 1.01e-02 | 1.17e-02 | 1.20e-02 | 1.20e-02 | 1.19e-02 | genuine, but already at level 1 |

The two classes are the **same point** in lead-lag's level-2 tensor space, to 1.2e-15. No
classifier of any capacity can separate them there. The chance-level accuracies in the table
above are a consequence of this, not the evidence for it.

**Sweeping the density is what makes this decisive, and it is easy to get wrong.** On an
*irregular* grid lead-lag's level-2 class difference is 6.4e-02 at $n=15$, which looks like a
result. It is not one: it falls to 2.4e-04 by $n=645$, tracking $Q_{tm}$, which is exactly
antisymmetric between the classes ($\pm 1.80\text{e-}02$ at $n=15$ falling to
$\pm 6.07\text{e-}05$ at $n=645$). An effect that vanishes as the path is resolved is
discretisation residue, not structure. A single-density table would have reported it as
signal.

### What restores level-2 ordering, and the mechanism

For an augmentation carrying a monotone time channel, level 2 supplies exactly
$\int x_i\,\mathrm{d}t$ for each other channel $x_i$, plus the pairwise areas among the
non-time channels. Order information reaches level 2 **if and only if some added channel has
a class-dependent time average**, which requires a *causal, non-anticipating* channel —
reflecting the data does not reflect the output of a backward-looking operator.

The cumulative integral $M = \int m\,\mathrm{d}t$ is the clean case and the mechanism is
exact: level 2 then contains $S^{t,M} = \int (t - t_0)\,m\,\mathrm{d}t$, the first temporal
moment. Verified two ways (check `V4`/`V6`): against an independent quadrature computed
outside the signature code, agreeing to **1.1e-16**; and the class difference in $S^{t,M}$ is
**exactly half** the class difference in the plain depth-3 word $S^{tmt}$, measured ratio
**0.500000000**. The causal channel pulls the depth-3 ordering functional down to depth 2
verbatim, at 12 features instead of 14.

Paired-fold tests on identical splits: cumulative integral at depth 2 (12 features) against
the plain path at depth 3 (14 features) differs by $+0.0063$ (logistic, $p = 0.034$) and
$-0.0087$ (trees, $p = 0.052$) — the two straddle zero, which is the point; running maximum at
depth 2 differs by $+0.0050$ ($p = 0.41$) and $+0.0013$ ($p = 0.80$).
The causal depth-2 representations **match the plain depth-3 representation at lower cost**.
This is candidate-level: it is established on synthetic data with a known ground truth and
has not been tested on photometry.

The EWMA row is the instructive intermediate case and the two tables appear to disagree about
it: its level-2 class difference is genuine and stable at 1.19e-02, yet it scores 0.533 / 0.499
at depth 2 on F1. Both are correct. Genuine is not the same as accessible — the effect is
twenty times smaller than the cumulative integral's 6.00e-02 and is swamped by the per-object
amplitude jitter and observational noise that F1 carries and the diagnostic family does not.
It also leaks into level 1 (6.4e-03), so it is not cleanly a level-2 phenomenon. A
machine-precision separation is a statement about what the representation contains, not a
promise that a classifier can reach it through noise; reporting both measurements is what
makes the distinction visible.

**The delay embedding was predicted to work and does not**, for a reason worth recording. The
reflection $m_B(t) = m_A(1-t)$ maps the delay path to itself with the lead and lag coordinates
exchanged *and* the traversal reversed; each operation flips the sign of a Lévy area, so their
composition preserves it. The hysteresis loop is class-invariant. What matters is not
non-monotonicity but covariance with the reflection. The one delay that does separate,
$\tau = 0.20$, separates already at **level 1** (1.25e-01), because a lag comparable to the
event timescale leaves the delayed channel truncated by the observation window so that its
net increment becomes class-dependent — a window-edge effect, not an iterated-integral one.

### Controlling for dimension

Lead-lag at depth 2 has 20 features against the plain path's 6, so a gain could have been
capacity. The negative result cannot be, and four controls rule it out from both directions:

| Control | Features | Logistic | Boosted trees | What it rules out |
|---|---|---|---|---|
| plain depth 3 | 14 | 0.979 | 0.978 | the winner is **narrower** than the failing lead-lag arm |
| plain depth 4 | 30 | 0.994 | 0.990 | 14 < 20 < 30, and the plain path wins at both ends |
| plain depth 2 + Gaussian padding | 20 | 0.520 | 0.461 | width alone buys nothing at matched dimension |
| plain depth 2 + Gaussian padding | 12 | 0.523 | 0.463 | the causal channels' 12 features are not width |
| lead-lag depth 2, random projection | 6 | 0.545 | 0.514 | lead-lag has nothing to lose by compression |
| cumulative integral depth 2, random projection | 6 | 0.956 | 0.910 | its signal survives compression to the plain width |
| lead-lag depth 2 + Gaussian padding | 84 | 0.494 | 0.525 | widening to a winning arm's size does not help |

The sandwich is the argument: the plain path succeeds at 14 features and at 30, and lead-lag
fails at 20. No capacity objection is available. Symmetrically, the cumulative integral's
12-feature win survives random projection to 6 features (0.956), the exact width at which the
plain path is at chance (0.514), so its advantage is information rather than dimension.

### Positive controls, each of which could have failed

A negative result is worthless without evidence that the apparatus can detect the positive.
All three passed, and one had to be strengthened before it was decisive.

| Control | Requirement | Measured | Verdict |
|---|---|---|---|
| Unit circle Lévy area | must be $\pm\pi$ | $\pm 3.141587$ against $\pi = 3.141593$ | pass |
| F5 orientation loop $(m_g, m_r)$, no time channel, traversed both ways | level 2 must reach $\sim 1.0$ | 1.000 / 0.995 at 6 features; magnitude-only at 0.488 | pass |
| F4 area differing (peaks swapped **and** area unequal) | plain level 2 **must** separate | 0.943 / 0.940 at 6 features | pass |
| F6 roughness differing (identical profile, identical ordering, extra interior jitter) | lead-lag level 2 must separate, plain must not | lead-lag 0.820 / 0.815; plain 0.478 / 0.448 | pass |

F5 is the check that the implementation sees orientation at all: when *neither* coordinate is
monotone the Lévy area is a real signed area and level 2 solves the task outright. The
degeneracy for time-augmented paths is geometric, not a bug.

F4 confirms the account of what level 2 contains. It also needed strengthening, and the reason
is recorded rather than tuned away: the level-2 entry is
$S^{mt} = \int (m - m_0)\,\mathrm{d}t$ and $m_0$ is a single noisy observation, so the
basepoint contributes a scatter of order $\sigma_{\text{noise}} \times \Delta t$. At a 1.35
area ratio that scatter is comparable to the class difference and the control read only 0.743
with trees (0.753 with the linear model), which demonstrates nothing; at 2.0 it reads 0.940.
Both ratios are in the outputs.

F6 identifies what lead-lag actually buys: **quadratic variation, which is order-blind**. At
*matched* feature count, lead-lag of the magnitude alone (6 features) scores 0.808 where the
plain time-augmented path (6 features) scores 0.478. The jitter is applied to interior points
only, so that the net increment — signature level 1 — is untouched. In a discarded variant that
jittered the endpoints too, the plain path reached 0.615 at depth 1 purely because the variance
of the net increment differed between classes, and the control was correspondingly less
specific. That variant is not in the committed outputs; the number is recorded because it is
the reason for the design choice.

### The sparse regime

Thinning to a median of **12 observations** per object, the cadence at which T1 found
signatures stop paying:

| Representation | Features | Logistic | Boosted trees |
|---|---|---|---|
| plain depth 2 | 6 | 0.511 | 0.550 |
| lead-lag depth 2 | 20 | 0.531 | 0.533 |
| **cumulative integral depth 2** | 12 | 0.779 | 0.795 |
| **running maximum depth 2** | 12 | 0.799 | **0.835** |
| plain depth 3 | 14 | 0.789 | 0.823 |
| plain depth 4 | 30 | 0.830 | 0.865 |

The lead-lag depth-2 null survives thinning, as it must. The causal-channel advantage also
survives: running maximum at depth 2 with 12 features (0.835) matches plain depth 3 with 14
(0.823). The whole problem gets harder — nothing reaches 0.9 at depth $\le 3$ — but the
ordering of the representations is unchanged.

### A cautionary number for T1

Lead-lag's depth-2 "gain" over the plain path on family F1 is **+0.0250** for the linear model
and **+0.0275** for the trees. Neither survives a paired-fold test on identical splits
($p = 0.051$ and $p = 0.154$). This is on data where the two classes' level-2 tensors are
provably identical to 1.2e-15, so the true effect is exactly zero and these are noise.

That gain is the same size as the **+0.0246** T1 measured for lead-lag on real ZTF photometry
and reported as the transform's contribution. T1 never ran a paired-fold test on that
ablation, and the gain sat inside one per-fold standard deviation of both arms. The present
experiment cannot show that T1's gain is noise — different data, different task — but it does
show that a gain of exactly that size arises readily where there is provably nothing to find.
**Before any mechanism is proposed for T1's +0.025, a paired-fold test on identical splits
should establish that the effect exists.**

### Verification

Five checks, each written so that it can fail, with the failure condition stated in its
docstring. All passed on this run.

| Check | Measures | Result |
|---|---|---|
| `V1` construction | T1's classes have matched marginals and matched net change | pooled magnitude mean 0.18505 vs 0.18695, KS $p = 0.281$; net change $+0.00388$ vs $+0.00808$, Welch $p = 0.422$ |
| `V2` exact reflection | diagnostic family is exact, not approximate | grid symmetry error **0.0**; net change **exactly 0.0** in both classes; sorted magnitudes bit-identical; reversal equals peak swap to 6.7e-16 |
| `V3` closed form | the six-scalar identity for lead-lag depth $\le 2$ | level-1 error **0.0**, level-2 error **8.9e-16** |
| `V4`/`V6` mechanism | $S^{t,M}$ is the first temporal moment, and it is half the depth-3 $S^{tmt}$ | quadrature error **1.1e-16**, ratio **0.500000000** |
| `V5` orientation | the primitive sees signed area | unit-circle Lévy area $\pm 3.141587$ |

Checks that compare a measured number against a tolerance of 1e-12 invite the question of
whether they could ever exceed it, so each is also re-run against a deliberately broken
version of the thing it tests (`mutation_test`). Every one fails by ten to fourteen orders of
magnitude when broken, which is what earns them the right to be cited above:

| Check | Correct | Deliberately broken | Ratio |
|---|---|---|---|
| `V3`, cross-(co)variation term $Q_{tm}$ removed from the closed form | 8.9e-16 | 4.4e-02 | $5\times10^{13}$ |
| `V3`, $Q_{tt}$ and $Q_{mm}$ removed | 8.9e-16 | 2.0e+00 | $2\times10^{15}$ |
| `V4`, level-2 index convention transposed | 1.1e-16 | 8.2e-02 | $7\times10^{14}$ |
| `V2`, an arbitrary random grid instead of a symmetric one | 0.0 | 1.6e-01 | — |
| `V5`, a path monotone in one coordinate instead of a circle | $\pi$ | 0.167 | — |

### What this changes

Question 1 of this track is answered for the standard augmentation family, and the answer is
narrower than the question assumed. The right statement is not "ordering lives at depth 3" but:

> For a path with a monotone time channel, level 2 supplies exactly $\{\Delta t, \Delta m,
> \int m\,\mathrm{d}t\}$ and nothing else. Doubling the channels by lead-lag adds only the
> realised (co)variation, which is invariant under permutation of the increments and hence
> order-blind. Ordering reaches level 2 only through a causal, non-anticipating channel whose
> time average is class-dependent.

Three consequences. First, lead-lag at depth 2 is not a cheap route to order-sensitivity and
should not be used as one. Second, lead-lag's real contribution is quadratic variation, which
is worth having where roughness carries class information — the F6 family, and by extension
the variable-source fraction of a survey sample — but is not ordering. Third, there is a
cheaper route to the depth-3 ordering functional than depth 3: append the cumulative integral
and stay at depth 2, twelve features against fourteen. That last claim is candidate-level and
synthetic; testing it on photometry is the obvious next step, alongside the paired-fold test
that T1's lead-lag ablation still needs.

## Note

This track is not astronomy and is not aimed at an astronomy venue. Its output is an explanation for
T1's numbers and, if the result is sharp, a machine-learning workshop paper.
