# T2 — Meteoroid stream discovery in an openly licensed orbital archive

**Status: complete, negative.** The method works and there is nothing new in this archive.
Results below.

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

## The first version was wrong, and how it was caught

The first complete run produced 4,517 consensus groups, of which 3,727 matched no known
shower. Read naively that is a spectacular discovery rate. It is nothing of the kind, and
the audit that caught it is the most useful thing this track has produced so far.

Three diagnostics killed it:

| Check | What it showed |
|---|---|
| Sporadic fraction of unmatched groups | median **1.000** — the groups were made entirely of meteors GMN calls sporadic |
| Member counts | median 14, but maximum **5,804** in a 6,000-meteor window |
| Distinct showers behind matched groups | 790 matched groups but only **169** distinct showers — every stream counted about five times |

The largest "cluster" held 97% of its window. DBSCAN chains through a dense continuous
background: however small the neighbourhood radius, sufficient density links everything
into one connected component. Those objects were the sporadic complex itself, not streams
within it. Meanwhile overlapping windows re-counted the same structures, and the novelty
test — "is the most common label in this group the sporadic marker?" — filed any genuine
stream that GMN had mostly left unlabelled as new.

Publishing that number would have been a serious false-discovery claim. Four corrections
follow, and they are now the substance of the method.

**1. A physically valid null.** Shuffling orbital elements independently preserves every
marginal distribution but produces orbits that cannot reach Earth. A meteoroid is only
observable if its orbit crosses Earth's, so null orbits failing $q \le Q_\oplus$ and
$Q \ge q_\oplus$ are rejected and redrawn. Measured effect: 4.2% of naively shuffled orbits
are unreachable, against 0% of the real data. Without the constraint the null is dispersed
over orbital space no meteor can occupy, its pair distances inflate, and every threshold
derived from it is too permissive.

**2. Group-level significance.** Each surviving group is tested as an object rather than
trusted because a clustering algorithm emitted it. The statistic is a density excess: the
group's centroid and its own radius define a ball in orbit space, and the observed count
there is compared against what the null puts in the same ball. A chained background
component spans a huge volume at background density and scores near zero however many
members it holds.

**3. A radius ceiling.** Significance alone was not enough — a 4,893-member component of
radius 1.98 in $D_{SH}$ still passed at $z = 9.0$. Classical association thresholds sit at
0.05–0.2, so a group whose 80th-percentile member distance exceeds 0.30 is not a stream by
any accepted definition. This is deliberately conservative and it costs completeness: the
known shower GCM was rejected at radius 0.316. Losing real showers to a strict cut is the
right trade when the alternative is claiming false ones.

**4. Both IAU lists.** Novelty is checked against the established list (113 codes) **and
the working list of 787 candidate showers**. Checking only the established list would
manufacture novelty out of showers other people have already reported.

After these corrections the Perseid window yields one group of 3,707 members at $z = 44.8$
and radius 0.11, entirely PER-labelled, while the chained background is rejected.

## Result

Run over **2,146,868 GMN orbits**. The consensus clustering recovers **64 distinct established
showers**, which is 57% completeness against the established catalogue — the method works.

**Replacing the null was the decisive step.** Unmatched structures fell from **498 to 56**, an
8.9-fold reduction, with no change to known-shower recovery. That is what a false-positive
problem looks like when it is fixed rather than thresholded away. The recurrence filter, which
separated known from unknown by **4.5 points** under the permutation null, separates them by
**40.7 points** under the sideband null (78.2% against 37.5%) — the same statistic, a better
null underneath. With a null that cannot see streams, no statistic built on it can either.

Of 21 candidates surviving the recurrence filter, 8 have median per-year z ≥ 3 across ≥ 5
apparitions. Re-checked against every IAU entry at loose tolerance, **eight of eight are known
showers**:

| z | Shower | Separation |
|---|---|---|
| 13.0 | FSL February sigma-Leonids | 3.9° |
| 8.5 | ACP alpha-Cepheids | 9.6° |
| 8.4 | EUM epsilon-Ursae Majorids | 8.0° |
| 7.4 | EDR epsilon-Draconids | 1.3° |
| 7.0 | ALA alpha-Lacertids (est.) | 2.6° |
| 5.7 | JES June epsilon-Serpentids | 8.4° |
| 3.4 | TAH tau-Herculids (est.) | 10.9° |
| 3.3 | DMC Daytime mu-Cancrids | 1.8° |

At 1.3° and 1.8° there is no ambiguity. The strict tolerance rejected these because catalogue
Vg values come from different solutions than the one measured here — the epsilon-Draconids
differ by 13% in Vg, inside the tolerance, and were excluded by the solar-longitude cut
instead. **So the surviving false positives come from the matching step, not the clustering:**
the clustering finds real streams and the bookkeeping failed to recognise them.

**No new stream is claimed.** What stands is a verified implementation; a measured comparison of
three sporadic-background models, showing that the permutation null widely used in
threshold-based work cannot separate streams from sporadic structure while a sideband null of
real orbits with rotated nodes can; and a null result on discovery at 57% completeness.

Unlike the first version of this track's results, this is a negative that survived having its
own null model replaced.

## Method

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
