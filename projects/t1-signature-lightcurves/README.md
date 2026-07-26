# T1 — Path signatures for irregularly sampled astronomical light curves

**Status: in progress.** Design fixed; dataset selection and implementation under way. Results are
not yet available, and no performance claim appears in this file until the code that produced it is
in `src/`.

## Overview

A photometric light curve is not a vector. It is a finite sample of a path: observations arrive at
times set by weather, scheduling and the target's position in the sky, in whichever filter the
telescope happened to be using, with per-point uncertainties that vary by an order of magnitude
across a season. Nearly every machine-learning pipeline for this data begins by destroying that
structure — interpolating onto a regular grid, binning to a fixed cadence, or fitting a Gaussian
process and resampling from it — so that a convolutional or recurrent architecture can accept the
result.

The signature transform of rough path theory accepts the raw stream directly. This track asks
whether that theoretical advantage survives contact with real photometry, and where.

## Scientific background

The signature of a path $X:[0,T]\to\mathbb{R}^d$ is the collection of its iterated integrals,

$$S(X) = \left(1,\ \int_{0<t<T} dX_t,\ \iint_{0<s<t<T} dX_s\otimes dX_t,\ \dots\right),$$

with the $k$-th term an element of $(\mathbb{R}^d)^{\otimes k}$. Three properties matter here.

**Reparameterisation invariance.** $S(X)$ is unchanged by any increasing reparameterisation of time.
The signature encodes the *order and shape* of what happened, not the clock it happened on. For data
where the observation times are set by the weather rather than by the source, this is the correct
invariance to have, and it is exactly what interpolation-based pipelines have to learn from data
instead.

**Faithfulness.** By the uniqueness theorem (Hambly and Lyons), the signature determines the path up
to tree-like equivalence — informally, up to retracing. Nothing is thrown away except the
parameterisation.

**Universality.** Linear functionals of the signature approximate continuous functions on path space
uniformly on compact sets. A linear model on signature features is therefore a universal
approximator on paths, which turns feature engineering into a question of truncation depth rather
than of architecture.

Against this: the invariance is exact for the underlying continuous path, while what is available is
a noisy, sparsely sampled version of it. Whether the guarantees have force under real survey cadence
is an empirical question, and it is the question this track exists to answer.

## Prior art, stated honestly

Path signatures have one published astronomical application: **SigNova** (Arrubarrena, Lemercier,
Nikolic, Lyons and Cass, arXiv:2402.14892, 2024), which detects radio-frequency interference in
interferometric visibility streams by scoring signature features against a clean reference set,
validated on Murchison Widefield Array and HERA data. A follow-up from the same group,
*Novelty detection on path space* (arXiv:2512.03243), develops the underlying hypothesis-testing
framework. In adjacent physical science, path signatures have been combined with graph neural
networks for slow-slip earthquake analysis (arXiv:2402.03558).

No application to photometric light curves, variable-star or transient classification, survey broker
feature sets, gravitational-wave strain, or asteroseismology was found in a targeted audit
(see [`docs/00-research-scan.md`](../../docs/00-research-scan.md), Part III). The gap is specific
and narrow, which is what makes it worth attempting: SigNova establishes that the representation
survives real radio-astronomical noise, and the open question is whether it earns its place in
time-domain photometry against a mature incumbent stack.

## The baselines this must beat

A method claiming to handle irregular sampling natively is claiming an advantage over methods that
already handle it adequately. The comparison is therefore against, at minimum:

| Baseline | What it is | Why it is the fair comparison |
|---|---|---|
| `feets` / FATS variability features | Hand-crafted statistical and periodicity features | The incumbent in survey pipelines; cheap and strong |
| Gaussian-process interpolation plus features | Fits a GP, resamples, extracts features | The standard principled treatment of irregular sampling |
| MiniRocket | Random convolutional kernels, deterministic feature map | The sharpest comparison: also a cheap fixed feature map, so the contest is structured iterated integrals against random projections at matched cost |
| Astromer / ATAT-style embeddings | Self-supervised transformer representations | The modern high-capacity option, included for context on the accuracy ceiling |

Reporting only the cases where signatures win would make the result worthless. Every baseline is
implemented and reported, including where it wins.

## The falsifiable prediction

The structure-first argument makes a specific prediction rather than a general claim of superiority:

> The signature representation should gain most where sampling is sparsest and most irregular, and
> least where light curves are dense and near-periodic. Its advantage should be largest for classes
> distinguished by the *order* of features in the light curve rather than by their marginal
> statistics, because ordering is precisely what iterated integrals encode and what order-blind
> summary statistics discard.

Both halves are testable by ablation, and both can fail. If signatures win uniformly, including on
dense periodic data, the explanation is probably not the one claimed and the result needs a
different account.

## Data

Two samples, chosen to test opposite halves of the prediction below. Access recipes, with the live
test results and the one failure encountered, are in
[`docs/02-data-sources.md`](../../docs/02-data-sources.md).

**Primary — ZTF Bright Transient Survey.** 12,916 spectroscopically classified transients (SN Ia
7,894, SN II 1,736, CV 685, AGN 404, and a tail of rarer types), with per-epoch $(t, m, \sigma,
\text{band})$ photometry in $g$ and $r$ served by the ALeRCE broker without authentication. Sampling
is set by weather and survey scheduling; light curves are short and sparse. Spectroscopic labels are
the ground truth, kept strictly separate from ALeRCE's own machine classifications.

**Counter-test — Gaia DR3 epoch photometry.** 9,976,881 sources with official variability
classifications, three bands ($G$, $BP$, $RP$) sampled at genuinely different instants within each
transit, at roughly 6 kB per source. Periodic variables, where period is the physically meaningful
quantity.

The second sample is not a robustness check. It is the case the method should handle *badly*, and
measuring how badly is the point.

## Signature implementation

`src/signature.py` implements truncated signatures from scratch: each straight segment contributes
the tensor exponential of its increment, and segments combine by Chen's identity. It is written
rather than imported because the theory track needs access to individual tensor levels, and because
the standard C++ backends do not install reliably across current Python versions — `iisignature`
fails to build on Python 3.12.

Correctness is verified by `src/verify_signature.py`, which checks agreement with three independent
libraries — `esig`, `signax`, and `roughpy` (the last from the group behind SigNova) — and four
structural identities: reparameterisation invariance, level 1 equalling the total increment, the
shuffle identity $S_1 S_2 = S_{12} + S_{21}$, and the vanishing of the signature of a path
concatenated with its own reversal. All sixteen checks pass, with agreement at the $10^{-13}$ level.

```bash
python3 src/verify_signature.py    # 16/16 checks
```

## Planned method

1. **Ingest** raw per-epoch photometry as $(t,\,m,\,\sigma,\,\text{band})$ tuples, with quality flags
   applied and no interpolation, binning or resampling at any stage of the signature path.
2. **Path construction.** Build the multi-band path, comparing basepoint, time-augmented and lead-lag
   augmentations. The lead-lag transform is what makes quadratic variation visible to the signature,
   which matters for stochastically variable sources.
3. **Signature features** at truncation depths 1 to 4, using log-signatures where the dimension
   demands it.
4. **Classification** with a deliberately simple linear model and a gradient-boosted tree, so that
   the representation rather than the classifier is on trial.
5. **Ablation over sampling.** Degrade the cadence systematically — random thinning against
   structured, survey-realistic gap patterns at matched density — and measure how each method
   degrades. This ablation, not the headline score, is the actual result.
6. **Uncertainty handling.** Photometric errors have no natural place in the signature construction.
   Options to be tested include error-weighted path construction and Monte Carlo perturbation of the
   input path. This is a genuine weakness of the approach and is treated as such.

## Risks recorded in advance

- **The baselines may simply win.** `feets` features and MiniRocket are strong, and the incumbent
  stack has absorbed a great deal of domain knowledge. A null result is a real possibility and would
  be reported as the headline.
- **Truncation may erase the theoretical advantage.** Universality is a statement about the full
  signature; at depth 3 or 4 the representation may amount to a set of quadratic and cubic
  statistics with no advantage over cheaper equivalents. T3 exists to measure this.
- **Multi-band paths are ill-posed.** Observations in different filters are not simultaneous, so
  building a single multi-dimensional path requires a choice about how to combine channels that
  were never sampled together. Different choices may dominate the result.
- **Reparameterisation invariance may be actively harmful.** Period is physically meaningful for
  variable stars, and a representation invariant to time reparameterisation discards it unless time
  is explicitly re-introduced as a channel. Where the invariance is wrong, it must be broken
  deliberately, and that decision must be reported rather than buried in a preprocessing step.

## References

1. Chevyrev, I. and Kormilitzin, A. (2016). *A Primer on the Signature Method in Machine Learning.* arXiv:1603.03788
2. Hambly, B. and Lyons, T. (2010). *Uniqueness for the signature of a path of bounded variation and the reduced path group.* Annals of Mathematics 171, 109.
3. Arrubarrena, P., Lemercier, M., Nikolic, B., Lyons, T. and Cass, T. (2024). *Novelty Detection on Radio Astronomy Data using Signatures.* arXiv:2402.14892
4. Dempster, A., Schmidt, D. F. and Webb, G. I. (2021). *MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification.* arXiv:2012.08791
