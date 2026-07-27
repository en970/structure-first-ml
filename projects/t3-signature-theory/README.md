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

## Note

This track is not astronomy and is not aimed at an astronomy venue. Its output is an explanation for
T1's numbers and, if the result is sharp, a machine-learning workshop paper.
