# T4 — Plate topology: a century of sky as image structure

**Status: closed, negative.** Four experiments, no pipeline built. The premise survived
three of them and failed the one it depended on: persistence diagrams of photographic plates
distinguish one patch of sky from another, but do not distinguish a patch of sky from its own
past — not even across the eruption of one of the brightest novae on record. The track closes
with that recorded.

## The opening

The Harvard plate collection covers the sky from the 1880s to the 1990s — 429,274 plates,
digitisation completed in 2024 — and carries something no modern survey has: **a
century-long time axis on the same patches of sky**.

The gap is narrow and evidenced. Of twenty arXiv papers using DASCH, **eighteen extract
point-source light curves**, and the remaining two touch plate imagery only inside the
reduction pipeline's own defect handling. Nobody appears to treat the plates as 2-D images,
or a plate series as an $(x, y, \text{epoch})$ object, and analyse their structure directly.

## Why the mathematics is cheap here and was not for Gaia

This repository already rejected a persistent-homology candidate. On Gaia phase-space point
clouds, degree-1 Vietoris–Rips persistence is intractable at survey scale — Ripser's own
documentation puts $H_1$ beyond practicality above roughly $10^3$ points, and $10^5$ points
in 5–6 dimensions sits at the frontier of specialised tools. That killed the idea, and the
rejection is recorded in [`docs/00-research-scan.md`](../../docs/00-research-scan.md).

On a 2-D image the natural construction is different in kind: a **cubical complex** over the
pixel grid with a sublevel-set filtration. Sort pixels by intensity, insert in order, track
when connected components ($H_0$) and loops ($H_1$) are born and die. It is near-linear. A
$1976 \times 2204$ plate is four million cells and takes seconds. The same mathematics that
was unusable on a point cloud is routine on a grid.

Feasibility is not speculative: **DRUID** (arXiv:2410.22508) already uses cubical-complex
persistence for source detection and deblending on optical and radio images. That narrows
the gap honestly — the computation is proven, and what is untouched is *structural
characterisation over time* rather than source finding.

## Data access, verified live

All endpoints tested on 2026-07-30 against `api.starglass.cfa.harvard.edu/public`, **no
authentication, no institutional account**:

| Call | Result |
|---|---|
| `POST /plates/search` | 9,036 plate identifiers near one position (plain strings, not records) |
| `GET /plates/p/{id}` | metadata; dates from 1896 onward; `catalog_exposures` carries `ctr_ra`, `ctr_dec`, `datetime` |
| `GET /plates/p/{id}/mosaic` | `bin_factor=16` → 4.8 MB, $1976\times2204$, real `RA---TAN` WCS |
| `POST /dasch/dr7/cutout` | calibrated $835\times835$ cutout at any RA/Dec, 1.44″/px |

Not every plate has a mosaic — `a02042` returns an empty list — so any survey must filter on
availability rather than assume it.

**Licence caveat:** the archive links to Harvard's terms of use, which were **not read**.
They must be checked before any plate imagery is redistributed. Nothing here redistributes
pixels.

## The data-product error that decided the first run

The first falsification run used 16×-binned whole-plate mosaics, and T1 failed with every
patch reporting a bright-pixel fraction of **exactly zero**. Measured on plate `a02115` over
the same field:

| Product | Background MAD | Fraction above 5 MAD |
|---|---|---|
| 16×-binned mosaic | **833.0** | **0.00000** |
| Calibrated cutout (1.44″/px) | **72.0** | **0.00522** |

Binning by sixteen collapses stellar profiles below one pixel and folds the plate's
vignetting gradient into the scatter, inflating the robust deviation elevenfold. The mosaic
is the right product for *locating* a plate and the wrong one for *analysing structure* on
it. Had this not been checked, the run would have read as a failure of the method rather
than of the product choice.

## Falsification results

Cutout-based, 36 patches from 12 plates, `src/falsify_grain.py`,
`outputs/falsify_grain.json`.

### T1 — does persistence respond to source content? **Yes, on one statistic only.**

Patches ranked by measured source fraction, then compared:

| | Source-poor | Source-rich |
|---|---|---|
| Source fraction | 0.0022 – 0.0050 | 0.053 – 0.073 |
| **Max persistence** | **715** | **2295** |
| Total persistence | 395,376 | 381,511 |
| Significant features | 496 | 418 |

Source content separates by a factor of twenty, and **maximum persistence tracks it by a
factor of 3.2**. But total persistence and the count of significant features do *not* — they
are flat or slightly inverted. Only one summary statistic carries the signal, which means
the signal is real but narrow, and a careless choice of summary would have missed it or
reported the opposite.

### T3 — can persistence tell a real field from a pixel-shuffled one? **Yes.**

| | Real | Shuffled |
|---|---|---|
| Components | 5,779 | 7,316 |
| Total persistence | 526,917 | 1,114,570 |
| Max persistence | 1,853 | 1,820 |

Bottleneck distance 643, which is 0.347 of the diagrams' own persistence scale. Spatial
coherence reduces the component count by a quarter and halves total persistence.

**A registered expectation that was wrong.** The original criterion required the *real* field
to show *longer*-lived features. That is backwards, and the data says so. Shuffling preserves
the histogram, so every bright pixel survives but is scattered into isolation — and an
isolated bright pixel on a flat background is a long-lived component. Real sources are
spatially coherent and merge into fewer components. Lifetime direction is not the
discriminant; **separability** is. The criterion was restated accordingly, and the original
expectation is left on the record, because a criterion rewritten after seeing the data has to
show its working.

### T2 — does the same sky position give consistent topology across plates? **Technically, barely.**

| | Median bottleneck |
|---|---|
| Within field, different plates (15 pairs) | 1,157 |
| Between different fields (36 pairs) | 1,183 |

The difference is **2.2%**. This passes the stated test and should not be treated as a
result. Cutouts are requested by RA and Dec, so this is genuinely WCS-aligned rather than a
pixel-window proxy — which makes the near-absence of separation more informative, not less.

**What it means.** Plate-to-plate variation — emulsion, exposure time, limiting magnitude,
development — is comparable to or larger than the difference between one patch of sky and
another. Any comparison across epochs therefore needs **plate-level normalisation** before
topology is computed, not after. Without it, the century-long time axis that motivates this
whole track is unusable.

## The normalisation gate: passed, and it refuted my own argument

`src/normalisation_gate.py`, 14 plates spanning 1896–1899, WCS-aligned cutouts,
`outputs/normalisation_gate.json`.

| Normalisation | Within field | Between fields | Relative separation |
|---|---|---|---|
| none | 1038.0 | 1377.0 | +0.327 |
| **zscore** | 9.73 | 15.84 | **+0.628** |
| rank | 0.1221 | 0.1207 | **−0.011** |

Threshold was fixed at 0.15 **before** the run. The gate passes on z-score normalisation,
which nearly doubles the separation over raw pixels.

**Two things I had written down were wrong.**

*First, the rank transform came last, not first.* The module argued at length that rank
normalisation was the principled choice, because sublevel-set persistence depends only on the
order in which pixels enter the filtration, making rank the canonical representative of the
monotone-transform class. That reasoning applies to the wrong object. Persistence is
invariant to monotone reparameterisation in the sense that the **shape** of the diagram —
which features exist and how they nest — is fixed by the ordering. But the diagram's
**coordinates**, the birth and death values, *are* the filtration values, and bottleneck
distance is computed on those coordinates. Forcing every patch to a uniform distribution on
$[0,1]$ makes all diagrams occupy the same narrow coordinate range whatever is in them. Rank
normalisation preserves the topology and destroys the metric. The absolute scale of the
filtration carries information, and the right correction removes nuisance offsets while
keeping that scale — which is what the z-score does.

*Second, the flatness that motivated this gate was a small-sample effect.* Unnormalised
separation here is **+0.327**, against +0.022 in the five-plate falsification run. Fourteen
plates instead of five was enough to open it. The earlier number was not a property of the
data, and the gate was built on a premise that was itself partly an artefact — though the
normalisation it prompted still turns out to be worth 0.30 on top.

Plate heterogeneity is real and large: across these 14 plates the robust scatter varies with
CV 0.48, the background level with CV 0.25, and the measured source fraction with CV **1.64**.
Exposure times run from 10 s to 300 s, and source fractions from 0.003 to 0.218 — a factor of
seventy. That is the nuisance z-score normalisation removes.

## The temporal gate: failed, and the track closes

`src/temporal_gate.py`, GK Persei (Nova Persei 1901), 14 plates from 1900–1904 bracketing the
February 1901 eruption, control field 0.8° away on the same plates, 91 pairs each.

| | Median | Mean | Std | Min |
|---|---|---|---|---|
| Target field (contains GK Per) | **1.944** | 7.090 | 9.169 | 0.644 |
| Control field | **3.234** | 6.348 | 7.485 | 0.000 |

Change signal **−0.399** against a threshold of +0.314 fixed before the run. **The gate
fails**, and it fails in the wrong direction: the field containing one of the brightest novae
on record is *more* self-consistent across epochs than a field with no such source.

**An audit of the test itself, which does not rescue it.** The distributions are extremely
skewed — standard deviation comparable to the mean — and median and mean point opposite ways
(the target's mean is higher, 7.09 against 6.35, while its median is lower). The control also
contains pairs at exactly 0.000, meaning some cutouts produced identical or empty diagrams.
So the median-based criterion is fragile on this data, and a different summary statistic
would have given a different verdict.

That fragility is a reason to distrust the number, not a reason to overturn it. The criterion
was fixed in advance precisely so it could not be renegotiated after seeing the result, and
nothing in the audit suggests the target field separates its own epochs *better* than the
control — only that the margin is noisier than a single median conveys. A method whose
verdict flips with the choice of summary statistic is not detecting a nova that brightened by
thirteen magnitudes.

**Conclusion: the topology distinguishes places, not times.** Gate 1 established that
z-score-normalised diagrams separate one field from another (0.628), and that separation is
evidently driven by which stars sit in a field — a property that does not change with epoch.
Gate 2 asked the question the track actually depends on and got no signal.

## Verdict, and what would kill this

The premise survives: persistence on calibrated plate cutouts does see astrophysical
structure rather than emulsion grain. But it survives on one statistic, and the
epoch-comparison test — the one the track's central idea depends on — is essentially flat.

**The track closes.** Gate 1 passed and gate 2 failed, which is the informative combination:
the method works and the premise does not. Whole-field persistence diagrams are dominated by
the population of sources in a field, and a single nova — even one that brightened from
magnitude 13 to 0.2 — is one object among thousands and does not move the diagram.

**What it would take to revive it, and why that is a different project.** The failure points
at scale, not at topology. A method that localises the comparison — persistence computed on
small windows around individual candidate positions rather than on whole 256-pixel fields —
might see a nova, because the changing object would no longer be diluted. But that is a
matched-filter search with topological features, not the structural-characterisation idea
this track was opened to test, and it would need its own premise test before any pipeline.

**What stands.** A verified DASCH image-access module, a falsification suite that caught a
data-product error before it was mistaken for a method failure, a measured comparison of
three normalisations that refuted the argument written for the principled one, and a clean
negative on the temporal premise obtained in four experiments rather than a built pipeline.

## Reproducibility

```bash
cd projects/t4-plate-topology
python3 src/dasch_access.py      # smoke test: search, metadata, mosaic
python3 src/falsify_grain.py     # the three falsification tests
```
