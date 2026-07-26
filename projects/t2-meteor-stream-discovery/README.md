# T2 — Meteoroid stream discovery in an openly licensed orbital archive

**Status: skeleton.** This directory contains a design and an experimental protocol. No results yet.
Work begins after T1 reaches a reportable state.

## Overview

The discovery track. Where T1 asks whether a mathematically motivated representation outperforms a
mature baseline, T2 asks a plainer question: is there anything in this archive that nobody has
looked for?

The Global Meteor Network is a distributed array of low-cost video cameras operated largely by
amateurs, publishing daily trajectory and orbit solutions under a CC-BY-4.0 licence. Each detected
meteor yields a heliocentric orbit — semi-major axis, eccentricity, inclination, argument of
perihelion, longitude of the ascending node — together with radiant coordinates, geocentric
velocity, and an absolute magnitude estimate. The archive is measured in gigabytes of tabular data
rather than terabytes of imagery, and a 2026 review characterised the machine-learning work on it as
still consisting of isolated experiments **[reported — to be verified against the primary source
before this file leaves skeleton status]**.

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

- **The archive may be better mined than the secondary literature suggests.** The saturation claim
  is unverified. The first task is a proper literature check, and if the answer is that this is
  well-trodden ground, the track is abandoned and the finding recorded.
- **Consensus across metrics may collapse to the intersection of their agreements**, which is likely
  to be exactly the set of already-catalogued showers. That would be a null result, and it would be
  reported as one.
- **Selection effects are severe.** Camera coverage is geographically uneven, sensitivity varies by
  station and by night, and the detection probability depends on radiant elevation and geocentric
  velocity. A cluster that tracks camera coverage rather than orbital structure is a false positive,
  and the analysis must be able to tell the difference.
