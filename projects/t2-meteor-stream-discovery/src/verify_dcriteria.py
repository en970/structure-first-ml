"""Verify the D-criteria implementations against analytic special cases.

No reference implementation is assumed. Instead, each criterion is checked where its
formula collapses to a closed form:

  identity      D(x, x) = 0 for every criterion, on random orbits
  symmetry      D(a, b) = D(b, a) for every criterion, on random orbits
  e-only        orbits differing only in eccentricity:
                    D_SH = |de|,  D_H = |de|,  D_D = |de| / (e1 + e2)
  q-only        orbits differing only in perihelion distance:
                    D_SH = |dq|,  D_H = D_D = |dq| / (q1 + q2)
  i-only        orbits differing only in inclination:
                    D_SH = D_H = 2 sin(di / 2); for D_D the closed form di/pi holds
                    only with the perihelion on the line of nodes (peri = 0), because
                    tilting the plane otherwise moves the perihelion direction itself
  peri-only     coplanar orbits (i = 0) differing only in argument of perihelion:
                    D_SH = D_H = e * 2 sin(dw / 2)
  U-only        D_N with only the speed differing: D_N = |dU|

plus a physical sanity check: two Geminid-like orbits must be close under every
criterion, and a Geminid against a Perseid must be far, with the conventional 0.1-0.2
association thresholds sitting between the two.

Run:  python3 src/verify_dcriteria.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dcriteria import d_d, d_h, d_n, d_sh  # noqa: E402

ATOL = 1e-12
results: list[bool] = []


def scalar(x) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def check(name: str, got, want: float, atol: float = ATOL) -> None:
    got = scalar(got)
    ok = abs(got - want) < atol
    results.append(ok)
    print(f"  {name:44s} got={got:.10f} want={want:.10f} {'OK' if ok else 'FAIL'}")


def check_true(name: str, cond: bool) -> None:
    results.append(bool(cond))
    print(f"  {name:44s} {'OK' if cond else 'FAIL'}")


def main() -> int:
    rng = np.random.default_rng(20260727)

    # random but physical orbits
    n = 40
    q = rng.uniform(0.05, 1.3, n)
    e = rng.uniform(0.1, 0.97, n)
    i = rng.uniform(0.0, 170.0, n)
    node = rng.uniform(0.0, 360.0, n)
    peri = rng.uniform(0.0, 360.0, n)

    print("identity and symmetry on random orbits")
    for name, fn in (("d_sh", d_sh), ("d_d", d_d), ("d_h", d_h)):
        m = fn(q, e, i, node, peri, q, e, i, node, peri)
        # tolerance 1e-6, not machine epsilon: d_d computes arccos of a cosine that is
        # 1 - O(eps) on identical orbits, and arccos amplifies that to O(sqrt(eps)).
        # Against association thresholds of ~0.1 this is irrelevant, but it is a real
        # property of the formula and the tolerance records it.
        check_true(f"{name} identity: max diag = {np.max(np.abs(np.diag(m))):.2e}",
                   np.max(np.abs(np.diag(m))) < 1e-6)
        check_true(f"{name} symmetry: max |M - M^T| = {np.max(np.abs(m - m.T)):.2e}",
                   np.max(np.abs(m - m.T)) < 1e-9)

    print("\nanalytic special cases")
    # e-only
    got = scalar(d_sh([0.9], [0.5], [30.], [100.], [200.],
                     [0.9], [0.7], [30.], [100.], [200.]))
    check("d_sh e-only = |de|", got, 0.2)
    got = scalar(d_h([0.9], [0.5], [30.], [100.], [200.],
                    [0.9], [0.7], [30.], [100.], [200.]))
    check("d_h e-only = |de|", got, 0.2)
    got = scalar(d_d([0.9], [0.5], [30.], [100.], [200.],
                    [0.9], [0.7], [30.], [100.], [200.]))
    check("d_d e-only = |de|/(e1+e2)", got, 0.2 / 1.2)

    # q-only
    got = scalar(d_sh([0.8], [0.6], [30.], [100.], [200.],
                     [1.0], [0.6], [30.], [100.], [200.]))
    check("d_sh q-only = |dq|", got, 0.2)
    got = scalar(d_h([0.8], [0.6], [30.], [100.], [200.],
                    [1.0], [0.6], [30.], [100.], [200.]))
    check("d_h q-only = |dq|/(q1+q2)", got, 0.2 / 1.8)
    got = scalar(d_d([0.8], [0.6], [30.], [100.], [200.],
                    [1.0], [0.6], [30.], [100.], [200.]))
    check("d_d q-only = |dq|/(q1+q2)", got, 0.2 / 1.8)

    # i-only
    di = 20.0
    want_sh = 2.0 * np.sin(np.deg2rad(di) / 2.0)
    got = scalar(d_sh([0.9], [0.6], [10.], [100.], [200.],
                     [0.9], [0.6], [30.], [100.], [200.]))
    check("d_sh i-only = 2 sin(di/2)", got, want_sh)
    got = scalar(d_h([0.9], [0.6], [10.], [100.], [200.],
                    [0.9], [0.6], [30.], [100.], [200.]))
    check("d_h i-only = 2 sin(di/2)", got, want_sh)
    # For d_d the i-only closed form needs the perihelion ON the line of nodes
    # (peri = 0): otherwise tilting the plane moves the perihelion direction itself
    # (sin beta = sin i sin peri) and the theta term is legitimately nonzero. The first
    # version of this test used peri = 200 deg and flagged the implementation as wrong;
    # a hand computation showed the implementation was right and the expectation wasn't.
    got = scalar(d_d([0.9], [0.6], [10.], [100.], [0.],
                    [0.9], [0.6], [30.], [100.], [0.]))
    check("d_d i-only (peri=0) = di/pi", got, np.deg2rad(di) / np.pi)

    # peri-only, coplanar
    dw = 30.0
    ecc = 0.6
    want = ecc * 2.0 * np.sin(np.deg2rad(dw) / 2.0)
    got = scalar(d_sh([0.9], [ecc], [0.], [0.], [100.],
                     [0.9], [ecc], [0.], [0.], [130.]))
    check("d_sh peri-only (i=0) = e*2sin(dw/2)", got, want)
    got = scalar(d_h([0.9], [ecc], [0.], [0.], [100.],
                    [0.9], [ecc], [0.], [0.], [130.]))
    check("d_h peri-only (i=0) = e*2sin(dw/2)", got, want)

    # D_N: identity, symmetry, U-only
    u = rng.uniform(0.5, 3.0, n)
    ct = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 360.0, n)
    lam = rng.uniform(0.0, 360.0, n)
    m = d_n(u, ct, phi, lam, u, ct, phi, lam)
    check_true(f"d_n identity: max diag = {np.max(np.abs(np.diag(m))):.2e}",
               np.max(np.abs(np.diag(m))) < 1e-9)
    check_true(f"d_n symmetry: max |M - M^T| = {np.max(np.abs(m - m.T)):.2e}",
               np.max(np.abs(m - m.T)) < 1e-9)
    got = scalar(d_n([1.0], [0.3], [40.], [120.], [1.4], [0.3], [40.], [120.]))
    check("d_n U-only = |dU|", got, 0.4)

    # physical sanity: Geminid pair close, Geminid vs Perseid far
    print("\nphysical sanity (association thresholds ~0.1-0.2)")
    gem1 = ([0.140], [0.896], [23.3], [261.3], [324.4])
    gem2 = ([0.145], [0.890], [23.8], [261.8], [324.9])
    per = ([0.949], [0.905], [113.1], [139.4], [150.5])
    for name, fn in (("d_sh", d_sh), ("d_d", d_d), ("d_h", d_h)):
        close = scalar(fn(*gem1, *gem2))
        far = scalar(fn(*gem1, *per))
        check_true(f"{name}: Geminid pair {close:.4f} < 0.1 < Perseid {far:.4f}",
                   close < 0.1 < far)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
