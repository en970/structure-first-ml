# structure-first-ml

**Live reports (GitHub Pages):** https://en970.github.io/structure-first-ml/

An independent research repository on *structure-first* machine learning for physical data.
The organising question is not "which architecture scores highest" but "what mathematical
object is this measurement actually an element of, and what representation respects that
structure".

## Guiding principle

Most machine-learning pipelines coerce data into the shape the model expects: irregular
observations are interpolated onto a regular grid, point clouds are voxelised, curved domains
are flattened, and conserved quantities are left to be learned by accident. Each of those steps
discards structure that was present in the measurement and introduces artefacts that were not.

The alternative pursued here is to reverse the order of reasoning:

> Identify the mathematical object the data belongs to — a path, a point cloud, a section of a
> bundle, an orbit of a dynamical system — then choose a representation with the invariances that
> object actually has, and only then attach a learning algorithm. Where the representation carries
> a theorem (universality, stability, equivariance, coverage), state the theorem and test whether
> its hypotheses hold on real data.

This is a methodological commitment, not a stylistic one: it produces falsifiable claims, because
a representation with a theorem behind it predicts *where* it should outperform a baseline and,
just as importantly, where it should not.

## Research tracks

The repository is organised into three tracks that differ in where they sit on the
mathematics-to-data axis. They are worked in order; a track marked *skeleton* contains a design
document and an experimental protocol, but no results yet.

| # | Track | Question | Status |
|---|-------|----------|--------|
| T1 | [Path signatures for irregularly sampled light curves](projects/t1-signature-lightcurves/) | Does a reparameterisation-invariant path representation beat interpolation-based features on real, gappy, multi-band photometry? | Result: no, but it is complementary |
| T2 | [Meteoroid stream discovery](projects/t2-meteor-stream-discovery/) | Can multi-criterion consensus clustering surface stream candidates in 2.15M GMN orbits? | Result: no new stream; the null model was the finding |
| T3 | [Truncation depth and sampling irregularity](projects/t3-signature-theory/) | Which augmentations restore order-sensitivity to signature level 2, and at what cost? | Result: lead-lag refuted, causal channels work |
| T4 | [Plate topology](projects/t4-plate-topology/) | Does sublevel-set persistent homology on century-old photographic plates see sky structure rather than emulsion grain? | Premise tested, survives weakly |

T1 is applied mathematics on real observations. T2 is discovery on an archive that the
machine-learning community has largely ignored. T3 is method theory, using controlled synthetic
data to explain the behaviour observed in T1. T4 moves from time series to images, and reuses a
method this repository had already rejected once — persistent homology, which is intractable on
high-dimensional point clouds and near-linear on a pixel grid.

## Documents

- [`docs/00-research-scan.md`](docs/00-research-scan.md) — the survey behind the choice of tracks: which archives are open and under-mined as of July 2026, and which mathematically rich methods remain unapplied to them.
- [`docs/01-publication-pathways.md`](docs/01-publication-pathways.md) — what an unaffiliated researcher can realistically publish, and the concrete submission artefact each pathway requires.
- [`docs/02-data-sources.md`](docs/02-data-sources.md) — access recipes for the open archives used here.

## Repository layout

```
structure-first-ml/
  README.md
  index.html                        landing page
  docs/                             research scan, publication pathways, data access
  projects/
    t1-signature-lightcurves/       path signatures on real photometry
    t2-meteor-stream-discovery/     orbital-archive discovery
    t3-signature-theory/            truncation depth and sampling theory
  reports/                          combined LaTeX technical report
```

Each track directory holds `README.md`, `src/` (runnable code), `outputs/` (figures and
machine-readable results), and `site/` (the HTML report).

## Relationship to space-ml-lab

[space-ml-lab](https://github.com/en970/space-ml-lab) is a sibling repository with a different
brief: it selects under-mined space-science archives and hunts for candidates in them, with the
method chosen for convenience. This repository inverts that emphasis — the method and its
mathematical guarantees are the object of study, and the archive is chosen because it stresses
those guarantees. Astronomical data is used throughout because irregular sampling, heteroscedastic
errors, and selection effects make it an honest test bed, not a sanitised benchmark.

## Data and licence

Code and written reports are released under the MIT Licence (see `LICENSE`). Analysed data remain
the property of their providers and are used under their open-data terms; sources are cited in each
track's report.

## Note on results

Results reported here are method-validating and candidate-level. Where a comparison against a
baseline is claimed, the baseline is implemented and reported in full, including the cases where it
wins. Negative results are kept rather than discarded.
