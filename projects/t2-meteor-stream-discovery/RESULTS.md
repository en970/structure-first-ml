# T2 results — a working method, a false-discovery rate I cannot yet defend

**Status: partial. No candidate is claimed as a discovery, and the reason is stated below
rather than buried.** Written 27 July 2026.

## What was built and verified

| Component | Verification |
|---|---|
| Four D-criteria ($D_{SH}$, $D_D$, $D_H$, $D_N$) | 23/23 checks: identity, symmetry, analytic special cases, Geminid/Perseid sanity |
| Significance machinery | 16/16 checks: `earth_crossing`, `density_excess`, `merge_overlapping` |
| IAU matching | Perseid radiant matches PER at 0.11°; a nonsense radiant matches nothing |
| Archive | 2,146,868 meteors, 91 monthly files, per-cut removals reported |
| Positive control | Perseid window returns one group, 3,707 members, $z = 44.8$, 100% PER |

The method runs, the components are individually verified, and it recovers known showers.

## What the sweep found

360° swept in overlapping 2° windows: 2,057 window-level groups passed $z \ge 5$, merging
to 1,249 distinct structures.

| Outcome | Count |
|---|---|
| Matched an IAU **established** shower | 188 |
| Matched only the IAU **working list** | 516 |
| Carried a GMN code but matched neither list | 47 |
| **Matched nothing** | **498** |
| Distinct established showers recovered | 64 of 113 |

Then, filtering the 498:

| Filter | Survivors |
|---|---|
| Density excess in ≥ 3 apparition years | 461 |
| Orbit-space deduplication at $D_{SH} < 0.10$ | 471 distinct orbits (barely merged — these are genuinely different orbits, not recounts) |
| Excess in ≥ 7 years **and** ≥ 40 gathered meteors | **57** |

## Why none of this is claimed as a discovery

**The arithmetic does not work.** The method recovers 64 of the 113 established showers —
a completeness of about 57%. If it simultaneously found 471 new ones, then real new streams
would outnumber the entire established catalogue several times over. Actual discovery rates
in this field are a handful per year across all networks. A method whose novel-detection
rate exceeds its known-recovery rate by sevenfold is reporting its own false-positive rate,
not the sky.

**Three filters were tried and each failed to separate the populations:**

| Filter | Known structures | Unmatched structures | Separates? |
|---|---|---|---|
| Recurrence: meteors present in ≥ 3 years | 99.5% | 97.2% | No |
| Recurrence: density *excess* in ≥ 3 years | 97.1% | 92.6% | Barely |
| Within 25° of a sporadic source | 48.9% | 38.6% | No — and in the wrong direction |

The first failure was a conceptual error worth recording: I counted whether meteors recur
in the candidate's orbital neighbourhood, but the sporadic background is there every year
too, so "meteors appear here annually" is true of essentially every region Earth samples.
Replacing presence with *excess* was the right correction and it still barely separated the
two populations.

The sporadic-source test was a hypothesis that the unmatched structures were concentrations
of the sporadic complex — helion, antihelion, apex — that an element-shuffled null cannot
model, because the shuffle destroys exactly the element correlations that produce them.
**That hypothesis was not supported**: unmatched structures sit *further* from the sporadic
sources than the known ones do. The explanation lies elsewhere and I have not found it.

## The most likely remaining explanations, in order

1. **The null is too weak.** Element-wise shuffling preserves each marginal but destroys the
   joint structure of the sporadic complex, so any real correlation in orbital elements —
   including ones that are not streams — registers as an excess. The literature's answer is
   a kernel-density model of the sporadic background (Shober & Vaubaillon 2024; Shober 2026),
   which is what a serious next attempt must implement. This is my leading candidate.
2. **IAU matching is too strict.** Tolerances of 8° in radiant, 15% in $V_g$ and 12° in solar
   longitude may miss listed showers whose catalogue entries are imprecise or whose several
   published solutions disagree. Loosening them would move structures from "unmatched" to
   "known" without any of them being new.
3. **The method genuinely detects many weak, real concentrations** that are neither
   established showers nor artefacts. This is the most flattering reading and therefore the
   one to trust least without independent evidence.

## What is genuinely established here

- A verified implementation of four orbital dissimilarity criteria and a consensus
  clustering procedure, with the positive control passing cleanly.
- A concrete, reproducible demonstration that **multi-criterion consensus plus a
  permutation null is not sufficient** to separate streams from sporadic structure at this
  archive's scale. That is a real methodological result, it is consistent with why the
  recent literature moved to KDE-based nulls, and it is worth more than a candidate list I
  cannot defend.
- A pipeline into which a better null can be dropped without rebuilding anything else.

## What would make this publishable

Implement the KDE sporadic-background model, re-run, and require that the known-recovery
rate exceed the novel-detection rate before any candidate is named. If after that a small
number of structures survive with excess in most apparition years, they become candidates —
and per the repository's standing decision, the submission file would be prepared for the
IAU Meteor Data Center and handed over, with the decision to submit resting with the author.

Until then the honest headline is the negative one, and it is the headline.
