# T2 — Meteoroid stream discovery in an openly licensed orbital archive

**Status: in progress.** Literature audit complete; data access verified live; pipeline under
construction.

## Overview

The discovery track. Where T1 asks whether a mathematically motivated representation outperforms a
mature baseline, T2 asks a plainer question: is there structure in this archive that the field's
own recommended method — which nobody has yet implemented — would surface?

The Global Meteor Network is a distributed array of low-cost video cameras operated largely by
amateurs, publishing trajectory and orbit solutions under a CC-BY-4.0 licence. Each detected
meteor yields a heliocentric orbit together with radiant coordinates, geocentric velocity with
per-quantity uncertainties, a Tisserand parameter, convergence angle, station count, and the GMN
pipeline's own IAU shower association. Access was verified live: monthly summary files from
December 2018 to the current month, 86 columns, 118,023 meteors in December 2025 alone
(~110 MB for that month), the whole archive laptop-scale.

## Literature status, audited honestly

An initial claim that machine learning had "barely been applied" to this archive was given to a
falsification scout and came back **partially false** — and the corrected picture is better for
this track than the naive claim was:

- **Shober (2026, ApJ, arXiv:2602.16845)** is the strongest prior art: DBSCAN plus KDE-derived
  sporadic-background null models and formal significance testing on 235,271 meteors from four
  networks (52% GMN), confirming the new "rock-comet" stream M2026-A1 at 5.3σ. Cross-*network*
  consensus, single dissimilarity metric ($D_N$).
- **Peña-Asensio & Ferrari (2025, AJ, arXiv:2507.01501)** ran HDBSCAN on CAMS (316,235 meteors,
  not GMN) and beat the classical look-up-table method on statistical coherence.
- **Shober & Vaubaillon (2024, A&A, arXiv:2404.08507)** computed four D-criteria in parallel on
  fireball data as a false-positive-rate estimator — parallel metrics, but not as a clustering
  acceptance test, and not on GMN.
- **Courtot, Shober & Vaubaillon (2025, arXiv:2507.19075)**, reviewing 40 papers, found most
  D-criterion use untested and **explicitly recommends combining multiple D-criteria with density
  clustering** — the method of this track, which no one has yet implemented.
- The GMN team's own operational discovery method (Šegon et al. 2026, eMetN, shower M2026-E1)
  remains classical: radiant/velocity windows plus single D-criteria, no clustering.

The niche, stated precisely: **GMN-native density clustering in which a candidate must survive
across several orbital dissimilarity criteria simultaneously ($D_{SH}$, $D_D$, $D_H$, $D_N$) to be
accepted — cross-metric stability as the detection statistic, calibrated against a
structure-respecting sporadic null.** Every ingredient has been separately demonstrated; the
combination is what the field's own review is asking for.

## Why this is a structure-first problem

Meteoroid stream identification is a clustering problem, but it is one in which the metric is not a
free choice. Two meteoroids belong to the same stream when they are on similar heliocentric orbits,
and "similar orbit" is a statement about celestial mechanics, not about Euclidean distance in a
five-dimensional parameter vector. The classical orbital dissimilarity criteria — Southworth–Hawkins
$D_{SH}$, Drummond $D_D$, Jopek $D_H$, and the Jenniskens variants — are competing attempts to
define that metric, and they disagree with each other in ways that propagate directly into which
streams get identified.

That disagreement is the opening. A clustering result that is stable across the family of orbital
metrics is a candidate; one that appears under a single choice of $D$ is an artefact of the metric.
Framing the problem this way makes metric sensitivity the object of study rather than a nuisance
parameter buried in a threshold.

A second structural fact: orbital elements are not independent coordinates on a flat space. The
angular elements are periodic, the eccentricity is bounded, and the physically meaningful notion of
proximity respects the geometry of the orbit space rather than the coordinate chart it happens to be
written in.

## Planned method

1. **Ingest and quality-cut.** Assemble the published trajectory and orbit solutions; apply
   convergence-angle, residual and station-count cuts to remove poorly constrained solutions. Record
   the cut fractions.
2. **Metric ensemble.** Compute pairwise dissimilarity under each of $D_{SH}$, $D_D$, $D_H$ and at
   least one modern variant, rather than committing to one.
3. **Consensus clustering.** Cluster under each metric independently, then retain only structures
   that survive across metrics. Stability across the ensemble is the detection statistic.
4. **Known-shower calibration.** Recover established showers from the IAU Meteor Data Center working
   list as positive controls, and quantify completeness and contamination against them. A method
   that cannot recover the Perseids has no standing to propose anything new.
5. **Sporadic-background null.** Construct a null distribution by randomising within the sporadic
   background — the sporadic complex is structured, not uniform, so the null must respect its
   helion, antihelion and apex sources. This is the step that determines whether any candidate is
   real, and it is where the effort belongs.
6. **Candidate assessment.** Any surviving candidate is checked against the IAU working list and the
   established-shower list before being described as new.

## Publication pathway

The IAU Meteor Data Center maintains the formal working list of meteor showers and has an
established process for candidate submission. A compact candidate list with a null-hypothesis
calculation is also within the scope of a Research Note of the AAS. See
[`docs/01-publication-pathways.md`](../../docs/01-publication-pathways.md).

## Risks recorded in advance

- ~~The archive may be better mined than the secondary literature suggests.~~ **Resolved by the
  audit above**: individual ingredients are demonstrated (most recently Shober 2026 on a
  GMN-dominated dataset), the multi-criterion consensus combination is not. The track proceeds
  with Shober's null-hypothesis machinery as the standard to meet, not as an unknown.
- **Consensus across metrics may collapse to the intersection of their agreements**, which is likely
  to be exactly the set of already-catalogued showers. That would be a null result, and it would be
  reported as one.
- **Selection effects are severe.** Camera coverage is geographically uneven, sensitivity varies by
  station and by night, and the detection probability depends on radiant elevation and geocentric
  velocity. A cluster that tracks camera coverage rather than orbital structure is a false positive,
  and the analysis must be able to tell the difference.
