"""Verify the significance machinery, not just the distance functions.

`verify_dcriteria.py` checks that the D-criteria are computed correctly. That is
necessary and not sufficient: the reported candidates depend just as much on three
functions that decide what counts as real, and none of them is a closed-form formula that
can be checked against an analytic special case. They are checked here against constructed
situations where the right answer is known by construction.

  earth_crossing      orbits with known geometry, classified by hand
  density_excess      a null-only window must score near zero; an injected compact
                      cluster must score high, and the score must rise with contrast
  merge_overlapping   groups with known sharing structure must merge into known components

The density test is the one that matters most. It is the sole defence against the failure
that produced 3,727 spurious candidates in the first version of this track, so a bug in it
would not announce itself -- it would look like a discovery.

Run:  python3 src/verify_significance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from consensus_cluster import (density_excess, earth_crossing,  # noqa: E402
                               merge_overlapping, physical_null)

results: list[bool] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append(bool(cond))
    print(f"  {name:52s} {'OK' if cond else 'FAIL'}{('  ' + detail) if detail else ''}")


def test_earth_crossing() -> None:
    print("earth_crossing on hand-classified orbits")
    # (q, e, should_cross, description)
    cases = [
        (0.14, 0.90, True, "Geminid-like: q inside, aphelion 2.66 AU"),
        (0.95, 0.92, True, "Perseid-like: q inside, large aphelion"),
        (0.10, 0.05, False, "tight circular inside Earth: aphelion 0.11 AU"),
        (1.50, 0.30, False, "entirely outside: q = 1.5 AU"),
        (1.00, 0.50, True, "q at 1 AU exactly"),
        (0.50, 0.99, True, "highly eccentric, aphelion ~99 AU"),
    ]
    q = np.array([c[0] for c in cases])
    e = np.array([c[1] for c in cases])
    got = earth_crossing(q, e)
    for (qq, ee, want, desc), g in zip(cases, got):
        check(f"q={qq:.2f} e={ee:.2f} -> {bool(g)}", bool(g) == want, desc)


def test_density_excess() -> None:
    print("\ndensity_excess on constructed windows")
    rng = np.random.default_rng(4242)

    # A background-only window: draw orbits from a broad distribution, then treat a
    # random subset as a "group". There is no real structure, so z must be near zero.
    n = 1500
    bg = pd.DataFrame({
        "q": rng.uniform(0.1, 1.0, n), "e": rng.uniform(0.4, 0.95, n),
        "i": rng.uniform(0, 60, n), "node": rng.uniform(0, 360, n),
        "peri": rng.uniform(0, 360, n),
    })
    bg = bg[earth_crossing(bg.q.to_numpy(), bg.e.to_numpy())].reset_index(drop=True)
    null_df = physical_null(bg, rng, size=800)

    fake = bg.sample(40, random_state=1)
    sig = density_excess(fake, bg, null_df)
    check(f"random subset of background scores low (z={sig['z']:.1f})",
          abs(sig["z"]) < 6.0, "no structure present")

    # An injected compact cluster on top of the same background must score high.
    centre = {"q": 0.55, "e": 0.72, "i": 24.0, "node": 133.0, "peri": 210.0}
    for spread, label in ((0.004, "tight"), (0.02, "loose")):
        clump = pd.DataFrame({
            k: rng.normal(v, spread * (60 if k in ("i", "node", "peri") else 1), 60)
            for k, v in centre.items()})
        window = pd.concat([bg, clump], ignore_index=True)
        s = density_excess(clump, window, null_df)
        check(f"injected {label} cluster scores high (z={s['z']:.1f}, r={s['radius']:.4f})",
              s["z"] > 10.0, "real structure present")

    # Contrast monotonicity, on a REAL solar-longitude window.
    #
    # Three versions of this test failed to measure anything, and the sequence is
    # instructive. The first placed the cluster far from any background, so the null
    # expected nothing in its ball and every variant returned exactly 48.0. The second
    # widened the cluster and still got n_exp = 0, because a synthetic background drawn
    # uniformly over five dimensions puts essentially no probability inside a small ball.
    # The third drew real orbits but sampled them from the whole archive, mixing every
    # solar longitude and dispersing them just as badly. Only a real narrow window
    # reproduces the production regime, where meteors concentrate on a thin Earth-crossing
    # manifold and n_exp is genuinely positive.
    #
    # In the n_exp > 0 regime the expectation is z ~ n_cluster / sqrt(rho V): thinning the
    # background raises the score. In the n_exp = 0 regime z collapses to n_obs and
    # thinning LOWERS it. The direction of the test reverses between regimes, which is
    # precisely why it must be run where production runs.
    archive = Path(__file__).resolve().parent.parent / "data" / "gmn_orbits.parquet"
    if archive.exists():
        real = pd.read_parquet(archive,
                               columns=["q", "e", "i", "node", "peri", "sol_lon",
                                        "iau_code"])
        win = real[(real.sol_lon >= 140.0) & (real.sol_lon < 142.0)]
        win = win.sample(min(4000, len(win)), random_state=3).reset_index(drop=True)
        cols = ["q", "e", "i", "node", "peri"]
        real_null = physical_null(win[cols], rng, size=2000)
        per = win[win.iau_code == "PER"][cols]

        if len(per) >= 30:
            s_dense = density_excess(per, win[cols], real_null)
            check(f"real Perseid group scores high (z={s_dense['z']:.1f}, "
                  f"n_exp={s_dense['n_exp']:.0f})",
                  s_dense["z"] > 10.0 and s_dense["n_exp"] > 1.0,
                  "production regime, n_exp positive")

            thin_bg = win[win.iau_code != "PER"].sample(frac=0.25, random_state=2)
            thinned = pd.concat([thin_bg[cols], per], ignore_index=True)
            s_thin = density_excess(per, thinned, real_null)
            check(f"thinning the background raises the score "
                  f"({s_thin['z']:.1f} vs {s_dense['z']:.1f})",
                  s_thin["z"] >= s_dense["z"] - 1e-9)
        else:
            print("  (Perseid contrast test skipped: too few PER in the window)")
    else:
        print("  (real-window contrast test skipped: run src/fetch_gmn.py first)")

    # And the regime itself is asserted, so the limitation cannot silently disappear.
    far = pd.DataFrame({k: rng.normal(v, 0.002 * (60 if k in ("i", "node", "peri") else 1), 30)
                        for k, v in {"q": 0.22, "e": 0.55, "i": 3.0,
                                     "node": 300.0, "peri": 40.0}.items()})
    s_far = density_excess(far, pd.concat([bg, far], ignore_index=True), null_df)
    check(f"empty-null regime collapses z to member count "
          f"(z={s_far['z']:.1f}, n_exp={s_far['n_exp']:.2f})",
          s_far["n_exp"] < 1.0 and abs(s_far["z"] - s_far["n_obs"]) < 1.5,
          "documented limitation, why the radius ceiling is separate")


def test_merge_overlapping() -> None:
    print("\nmerge_overlapping on known sharing structure")

    def grp(members, z, w0=0.0):
        return {"members": set(members), "window_start": w0, "window_end": w0 + 2.0,
                "significance": {"z": z}, "ra": 0.0, "dec": 0.0, "vgeo": 0.0,
                "sol_lon": w0, "gmn_code": None, "gmn_code_fraction": 0.0,
                "sporadic_fraction": 1.0, "q": 0.0, "e": 0.0, "i": 0.0,
                "peri": 0.0, "node": 0.0}

    # A-B share a member, C is disjoint: expect two components of sizes 4 and 2.
    groups = [grp("ab", 10.0, 0.0), grp("bc", 20.0, 1.5), grp("xy", 8.0, 100.0)]
    merged = merge_overlapping(groups)
    sizes = sorted(m["n_members"] for m in merged)
    check(f"two components with sizes {sizes}", sizes == [2, 3])

    # transitivity: A-B share, B-C share, A-C do not; all three must land together
    groups = [grp("ab", 5.0), grp("bc", 6.0), grp("cd", 7.0)]
    merged = merge_overlapping(groups)
    check(f"transitive chain merges into {len(merged)} component", len(merged) == 1,
          f"{merged[0]['n_members']} distinct members")

    # the representative kept must be the highest-z member of the component
    check(f"representative carries max z ({merged[0]['best_z']})",
          merged[0]["best_z"] == 7.0)

    # disjoint groups must never be merged
    groups = [grp(f"{i}{i}", 5.0, i * 10.0) for i in range(5)]
    merged = merge_overlapping(groups)
    check(f"five disjoint groups stay separate ({len(merged)})", len(merged) == 5)


def main() -> int:
    test_earth_crossing()
    test_density_excess()
    test_merge_overlapping()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
