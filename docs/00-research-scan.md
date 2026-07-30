# Research scan — what is open, what is under-mined, what is unapplied

*Compiled 26 July 2026. This document records the survey that led to the choice of tracks. It is
kept in the repository so that the reasoning is auditable, including the parts that turned out to be
wrong.*

Entries are marked **[verified]** where a primary source page was fetched and read during the scan,
and **[reported]** where the claim comes from secondary sources and has not yet been confirmed
against the archive itself. Nothing in this document should be cited without re-checking the
primary source; release schedules in particular move.

---

## Part I — Open archives, July 2026

The question asked was: which public space-science archives were released or substantially expanded
between mid-2024 and July 2026, are still under-mined by the machine-learning community, and can be
worked on a laptop or a free-tier Colab session?

| Archive | Status (July 2026) | ML saturation | Laptop-scale |
|---|---|---|---|
| Rubin / LSST alert stream | Public via community brokers since 2026-02-24 **[reported]** | Low on real alerts | Yes, filtered |
| Rubin DP1 / DP2 images | Data-rights restricted; public DR postponed to ~2028 **[reported]** | n/a | No — access-gated |
| Euclid Q1 | Public since 2025-03-19; DR1 scheduled 2026-11 **[reported]** | High for strong lensing, and **corrected 2026-07-30: also high for low-surface-brightness structure** | Yes, via TAP subsets |
| SPHEREx QR2 | Weekly public releases; IRSA and AWS Open Data **[reported]** | Low | Yes, per-tile |
| Gaia DR4 | Not yet released; announced for 2026-12-02 **[reported]** | Unmined by construction | Later |
| DESI DR1 | Public since March 2025, ~19M spectra | Medium; only a small subset mined | Yes, via SPARCL |
| JWST MAST (MRS / NIRSpec IFU cubes) | Ongoing public | Low for IFU cubes as a corpus | Yes, curated subset |
| CHIME/FRB Catalog 2 | Public 2026 **[reported]** | Medium for repeaters, low for sub-burst morphology | Yes |
| LoTSS DR3 (LOFAR) | Catalogues public 2026 **[reported]**; visibilities PB-scale | Low–medium | Catalogues only |
| GWOSC O4b | Public 2026-05-26 **[reported]** | High for glitches, low for overlapping-morphology separation | Yes |
| IceCube IceTracks-DR2 | Public 2026-05 **[reported]** | Low outside the collaboration | Yes |
| Global Meteor Network | Continuous, CC-BY-4.0 | Low — described as isolated experiments | Yes, GB-scale |
| TESS QLP | 9.1M light curves; new 200 s cadence FFIs | Medium overall, low on the new cadence | Yes |

The important negative result: **Rubin imaging is not available to an unaffiliated researcher.**
DP1 and DP2 are restricted to data-rights holders and the originally planned public DR1 was
cancelled; only the alert stream, distributed through community brokers, is genuinely open. Any
plan built on "just use LSST data" is not currently executable from outside the collaboration.

---

## Part II — Mathematically rich methods and where they have not been applied

The complementary question: which methods carry real mathematical content — a theorem, an
invariance, a guarantee — and remain unapplied to the archives above?

### Low saturation, laptop-scale, strong mathematics

**Path signatures / rough path theory.** The signature of a path $X:[0,T]\to\mathbb{R}^d$ is the
sequence of iterated integrals $\int dX_{i_1}, \iint dX_{i_1}dX_{i_2},\dots$, living in the tensor
algebra over $\mathbb{R}^d$. It is invariant under reparameterisation, faithful up to tree-like
equivalence, and universal in the sense that linear functionals of the signature approximate
continuous functions on path space. Standard in finance and clinical time series; see Part III for
the astronomy audit.

**Persistent homology.** A filtration of simplicial complexes over a point cloud, tracking the birth
and death of $k$-dimensional homology generators. The stability theorem bounds the bottleneck
distance between persistence diagrams by the perturbation of the input, which is exactly the
robustness property a noisy astronomical point cloud needs. Applied to the cosmic web; see Part III
for the stellar-stream audit.

**Koopman operator theory and DMD.** Lift a nonlinear system $x_{t+1}=F(x_t)$ to a linear operator
acting on observables, $(\mathcal{K}g)(x)=g(F(x))$, then approximate its leading spectrum from data.
Applied to SDO EUV imagery; no application found to in-situ solar-wind time series, where the
literature is dominated by black-box recurrent forecasters.

**Hamiltonian and Lagrangian neural networks.** Parameterise a scalar $H(q,p)$ and derive dynamics
from Hamilton's equations, integrated symplectically, so that energy conservation and
time-reversibility hold by construction. Demonstrated on toy systems; applications to real
astronomical N-body problems are sparse.

**Conformal prediction.** Distribution-free finite-sample coverage from exchangeability alone. Alert
brokers currently publish softmax scores with no coverage guarantee; a conformalised broker output
would be a cheap, mathematically honest improvement.

### High saturation — avoided

Spherical CNNs on the CMB, `celerite`-style Gaussian processes for transit and radial-velocity time
series, CNN strong-lens finders on Euclid, and supervised transient classifiers trained on ZTF are
all mature. Entering these means competing with well-resourced collaborations on their own ground.

---

## Part III — Falsification checks

Two claims were load-bearing for the track selection, so each was given to a separate scout with
instructions to disprove it.

### Claim: path signatures have never been applied to astronomical data

**Verdict: partially false, and the correction matters.**

One genuine application exists. Arrubarrena, Lemercier, Nikolic, Lyons and Cass, *Novelty Detection
on Radio Astronomy Data using Signatures* (arXiv:2402.14892, 2024) introduces SigNova, which
computes path signatures of radio visibility streams and scores novelty by Mahalanobis distance to a
clean reference set, validated on Murchison Widefield Array and HERA data. A follow-up methodology
paper from the same group, *Novelty detection on path space* (arXiv:2512.03243), cites radio
interference detection as its motivating application. The DataSig programme at Oxford and the Alan
Turing Institute lists astronomy as an application theme on the strength of this work.

What was *not* found, despite targeted searching: any signature-based method for photometric light
curves, variable-star or supernova classification, PLAsTiCC/ELAsTiCC, broker feature sets,
gravitational-wave strain, solar flare prediction, pulsar timing, or asteroseismology. Adjacent
physical sciences show one close application — path signatures with graph neural networks for slow
slip earthquakes (arXiv:2402.03558) — and none in particle physics or cosmology.

The revised, defensible statement is therefore: *path signatures have been applied once in radio
astronomy, to interference detection in visibility data; time-domain photometric classification is
untouched.* The prior art is acknowledged rather than hidden, and it strengthens the case, because
it establishes that the representation survives contact with real radio-astronomical noise.

The scan also fixed the baseline set that any honest comparison must include: hand-crafted
variability features (FATS, `feets`, the ALeRCE extractor), Gaussian-process interpolation and
augmentation, self-supervised transformer embeddings (Astromer, ATAT), and random convolutional
kernel methods (Rocket / MiniRocket). MiniRocket is the sharpest comparison, since it is also a
cheap deterministic feature map — random projections against iterated integrals.

### Claim: persistent homology has not been applied to Gaia phase space for stream detection

**Verdict: the gap is real, and the idea should still be rejected.**

No published application of persistent homology, Mapper, or any topological method to Gaia
phase-space data for stream or kinematic-substructure detection was found. Topological methods have
reached other astronomical data — the cosmic web (arXiv:2009.04819 and successors, all on gridded
density fields via cubical complexes rather than raw point clouds) and the CHIME/FRB catalogue
(arXiv:2311.03456) — but not Gaia kinematics.

Three findings nonetheless disqualify it as the first track:

1. **Computational infeasibility at the relevant scale.** Degree-1 Vietoris–Rips persistence is
   worst-case exponential in the number of points. Ripser's own documentation notes that $H_1$
   becomes impractical above roughly $10^3$ points without subsampling, and specialised tools
   (Dory, arXiv:2103.05608; giotto-ph, arXiv:2107.05412) exist precisely because $10^5$ points in
   4–6 dimensions remains at the frontier. A full-catalogue $H_1$ computation on Gaia is not a
   laptop exercise, and the landmark and sparse-Rips approximations that would make it tractable
   undercut the "template-free on raw phase-space points" claim that motivated it.

2. **The homology degree does not match the physics.** Stellar streams are thin, elongated,
   curve-like structures. Those are $H_0$ features — connectivity and merge-tree structure — not
   $H_1$ loops. The loop-detection machinery that makes persistent homology compelling for voids in
   the cosmic web is aimed at the wrong invariant here.

3. **Strong, directly competing prior art.** Via Machinae (arXiv:2104.12789), Via Machinae 2.0
   (arXiv:2303.01529, 102 all-sky candidates from Gaia DR2) and Via Machinae 3.0
   (arXiv:2509.08064, which discovered the Raritan and Passaic streams) already perform
   template-free, model-agnostic stream search on Gaia using normalising-flow density estimation
   plus a Hough transform, with no assumed Galactic potential. STREAMFINDER (arXiv:1804.11338)
   covers the matched-filter case, and HDBSCAN in integrals-of-motion space (arXiv:2410.06646)
   covers clustering. A topological method would have to explain what it adds to all of these
   before it could even be benchmarked.

Recording this rejection is the point of the exercise. The candidate looked strong on the two
criteria that are easy to check — novelty and mathematical richness — and failed on the two that
are easy to overlook: whether the invariant matches the physical structure, and whether the
computation fits the hardware actually available.

---

## Part IV — Why these three tracks

**T1, path signatures on light curves,** was chosen because the gap is real and narrow, the
mathematics is genuine rather than decorative, the data is free and abundant, the compute fits on a
laptop, and — decisively — the method's central claim is *testable*. Signatures should help most
where sampling is most irregular and least where a light curve is dense and near-periodic. That is a
prediction, and it can fail.

**T2, meteoroid streams,** was chosen as the discovery track because the Global Meteor Network is
openly licensed, small enough to hold in memory, essentially untouched by machine learning, and has
a real registry pathway for candidate streams. It is the closest thing to green field found in the
scan.

**T3, truncation depth and sampling irregularity,** exists because T1 will produce a number that
demands an explanation. A truncated signature keeps iterated integrals up to depth $m$, giving
$\sum_{k=0}^{m} d^k$ coefficients; how the useful information distributes across depths, and how that
distribution shifts as sampling becomes sparser and gappier, is a question that controlled synthetic
experiments can answer and real data cannot.

---

## Correction log

**2026-07-30 — Euclid low-surface-brightness structure is not an opening.** The table above
originally recorded LSB structure as less saturated than strong lensing, on the strength of a
first-pass scan. A later scout, briefed to find image-processing openings, found the opposite:
the Euclid Consortium runs its own Strong Lensing Discovery Engine (arXiv:2503.15324 and
2503.15325, 497 candidates over 63 deg²), and the LSB regime already carries a UDG
cross-survey domain-adaptation paper (arXiv:2605.13842), ERO dwarf-galaxy work
(arXiv:2405.13502) and tidal-stream analysis inside the footprint (arXiv:2411.09608). This is
a large, funded, actively staffed area, not a gap. An unaffiliated researcher is not first
here.

A related figure that circulated through the first scan — that Euclid lens-finder
completeness drops from about 92% on simulations to about 50% on real Q1 data — **could not be
traced to either consortium paper and is withdrawn as unverified.** It was never used in any
result, but it was quoted as motivation and should not have been without a source. The
underlying concern (a sim-to-real domain gap) remains plausible and is indirectly supported by
the pipeline needing deep learning, citizen-science review and expert vetting stacked together
to reach usable purity, but the numbers are not established.

The thinner sub-area, if any, is tidal-feature and intracluster-light *morphology* as opposed
to detection, where the scan found only pre-Euclid CNN work (arXiv:2404.06487) anticipating
Euclid rather than using Q1.

## Part V — On tooling

Anthropic's Claude Science workbench, announced 2026-06-30 **[reported]**, is configured for
genomics, single-cell biology, proteomics, structural biology and cheminformatics, and draws on
life-science databases. It carries no astronomy configuration, and no official connector exists for
MAST, SIMBAD, VizieR or any observatory archive. The work in this repository therefore uses ordinary
open-source tooling — `astroquery`, `numpy`, `scikit-learn`, `iisignature`/`signatory`, PyTorch —
with no dependence on a proprietary science platform. Anthropic's AI for Science credit programme
exists and is open on a rolling basis, but its eligibility text presumes institutional attachment
**[reported]**.
