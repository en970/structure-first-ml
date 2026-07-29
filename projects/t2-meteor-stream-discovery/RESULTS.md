# T2 results — the method works, and there is nothing new in this archive

**Status: complete, negative.** No new meteoroid stream is claimed. Every candidate that
survived the full filter chain turned out to be a known shower that a too-strict matching
tolerance had missed. Written 27–30 July 2026.

## Headline

| | first attempt | after the null was fixed |
|---|---|---|
| Distinct structures | 1,249 | **358** |
| Unmatched to any IAU list | 498 | **56** |
| Surviving multi-apparition recurrence | 461 | **21** |
| Strong subset (year-$z \ge 3$, ≥5 apparitions) | — | **8** |
| Of those, known showers found by loose re-check | — | **8 of 8** |
| Distinct established showers recovered | 64 of 113 | 64 of 113 |

The method recovers 57% of the established catalogue and, after correction, invents nothing.

## What was verified

| Component | Verification |
|---|---|
| Four D-criteria ($D_{SH}$, $D_D$, $D_H$, $D_N$) | 23/23: identity, symmetry, analytic special cases, Geminid/Perseid sanity |
| Significance machinery | 16/16: `earth_crossing`, `density_excess`, `merge_overlapping` |
| IAU matching | Perseid radiant matches PER at 0.11°; nonsense radiant matches nothing |
| Archive | 2,146,868 meteors, 91 monthly files, per-cut removals reported |
| Positive control | Perseid window: one group, 3,707 members, $z = 44.8$, 100% PER |

## The null model was the whole problem

The first sweep returned 471 unmatched structures against 64 recovered known showers — a
novel-detection rate seven times the known-recovery rate, which is a method reporting its
own false positives. The permutation null was named as the leading suspect and then
**measured** rather than assumed. A null has two jobs: predict roughly the observed count in
sporadic regions (calibration), and far fewer at a real stream (discrimination). Across the
Perseid, Southern Taurid and Geminid windows:

| Null model | Sporadic ratio (≈1 good) | Stream ratio (low good) | Separation |
|---|---|---|---|
| **sideband** — real orbits from adjacent solar longitudes, nodes rotated | **0.800** | 0.107 | **7.48×** |
| kde — joint density, bandwidth above stream scale | 0.369 | **0.072** | 5.12× |
| permutation — marginals only | 0.366 | 0.458 | **0.80×** |

The permutation null's separation is *below one*: it predicts about the same fraction at a
real shower as in empty background, so it cannot distinguish a stream from ordinary sporadic
structure at all. The reason is structural — the helion, antihelion, apex and toroidal
sources exist *because* orbital elements are correlated, and shuffling them independently
models a background that does not exist.

Two independent replacements agree at the streams (0.107 and 0.072), which is the check on
both. The sideband null needed one correction that failed in a way resembling success: with
no node rotation it predicted zero density at sporadic regions *and* zero at the Perseids,
which reads as perfect discrimination and is total collapse. A meteoroid is only observable
where its orbit crosses Earth's, so its ascending node is locked to the solar longitude of
the encounter; orbits borrowed from 18° away carry nodes 18° away, outside every $D_{SH}$
ball centred in the window.

**Replacing the null cut unmatched structures from 498 to 56 — an 8.9-fold reduction — with
no change to known-shower recovery.** That is what a false-positive problem looks like when
it is fixed rather than thresholded away.

## The recurrence filter only became a filter afterwards

A stream is a debris trail on a fixed orbit; Earth crosses it at the same solar longitude
every year. So a real stream should show a density excess in year after year, and a chance
alignment should not.

| Filter | Known structures | Unmatched | Separation |
|---|---|---|---|
| Meteors *present* in ≥3 years | 99.5% | 97.2% | 2.3 pts |
| Density *excess* in ≥3 years, permutation null | 97.1% | 92.6% | 4.5 pts |
| Density *excess* in ≥3 years, **sideband null** | **78.2%** | **37.5%** | **40.7 pts** |

The first row was a conceptual error: counting whether meteors recur near the candidate's
orbit measures nothing, because the sporadic background is there every year too. The second
row shows the corrected statistic still failing — with a null that cannot see streams, no
statistic built on it can either. Only the third row is a filter, and it is the same
statistic with a better null underneath.

## Why nothing is claimed

Of the 21 candidates surviving recurrence, 8 have a median per-year $z \ge 3$ across at
least 5 apparitions. Re-checking those 8 against every IAU entry at loose tolerance
(radiant 20°, $V_g$ 30%, no solar-longitude cut):

| Candidate | RA | Dec | $V_g$ | Nearest IAU shower | Separation |
|---|---|---|---|---|---|
| $z=13.0$, n=43 | 176.6 | +6.4 | 39.3 | FSL February σ-Leonids | 3.9° |
| $z=8.5$, n=22 | 340.0 | +64.2 | 13.2 | ACP α-Cepheids | 9.6° |
| $z=8.4$, n=10 | 205.8 | +67.6 | 16.5 | EUM ε-Ursae Majorids | 8.0° |
| $z=7.4$, n=38 | 309.6 | +72.0 | 20.5 | EDR ε-Draconids | **1.3°** |
| $z=7.0$, n=11 | 338.7 | +54.3 | 35.2 | ALA α-Lacertids (established) | 2.6° |
| $z=5.7$, n=65 | 238.3 | +12.8 | 12.0 | JES June ε-Serpentids | 8.4° |
| $z=3.4$, n=82 | 222.2 | +28.4 | 13.1 | TAH τ-Herculids (established) | 10.9° |
| $z=3.3$, n=13 | 123.8 | +23.7 | 22.3 | DMC Daytime μ-Cancrids | **1.8°** |

**Eight of eight are known showers.** At 1.3° and 1.8° there is no ambiguity. The strict
matching tolerance ($V_g$ within 15%, solar longitude within 12°) rejected them because
catalogue entries carry $V_g$ values from different solutions than the one measured here —
ε-Draconids differs by 13% in $V_g$, inside the tolerance, but was excluded by the
solar-longitude cut.

So the remaining false positives come from the **matching step, not the clustering step**.
The clustering finds real streams; the bookkeeping failed to recognise them.

## What this establishes

1. **A verified implementation** of four orbital dissimilarity criteria, consensus
   clustering, density-excess significance, and multi-apparition recurrence testing, with
   every component checked against constructed cases and known showers.
2. **A measured comparison of three sporadic-background models**, showing quantitatively
   that the permutation null used widely in threshold-based work cannot separate streams
   from sporadic structure, and that a sideband null of real orbits with rotated nodes does
   — a result that stands independently of whether anything new is found.
3. **A null result on discovery**, arrived at honestly: multi-criterion consensus clustering
   over 2.15 million GMN orbits surfaces no stream that is not already catalogued, at this
   sensitivity (57% completeness on the established list).

## What would change the answer

- **Loosen matching and re-run.** This would not produce discoveries; it would move
  structures from "unmatched" to "known" and shrink the candidate list further, which is
  the correct direction.
- **Push sensitivity rather than novelty.** Recovering more than 57% of the established
  list — by lowering `min_cluster`, widening the radius ceiling, or clustering on
  geocentric invariants ($D_N$) instead of osculating elements — is the interesting
  remaining direction, since it tests the method rather than the sky.
- **A larger archive.** GMN grows continuously; a rerun in a few years covers more
  apparitions, which is exactly what the recurrence filter rewards.

The honest headline is the negative one, and unlike the first version of this document, it
is now a negative that survived having its own null model replaced.
