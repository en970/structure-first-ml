"""Verify the from-scratch signature implementation in `signature.py`.

Two kinds of check are run.

Cross-library agreement: the same signature is computed with `esig`, `signax` and
`roughpy` -- three independent implementations, one of them (roughpy) from the group that
published the only prior astronomical application of signatures -- and compared
coefficient by coefficient. Any library that is not installed is skipped and reported as
such rather than silently passing.

Structural properties: four identities that must hold for any correct implementation,
checked without reference to another library.

  1. Reparameterisation invariance. Inserting extra vertices along existing segments
     changes the parameterisation but not the geometric path, so the signature must not
     move. This is the property the whole approach rests on.
  2. Level 1 equals the total increment, by definition of the first iterated integral.
  3. The shuffle identity S_1 S_2 = S_{12} + S_{21}, the simplest instance of the shuffle
     product that makes the signature a character on the shuffle algebra.
  4. A path concatenated with its own reversal is tree-like, so its signature is the
     identity element of the tensor algebra.

Run:  python3 src/verify_signature.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signature import signature  # noqa: E402

ATOL = 1e-10
CASES = [(2, 5, 3), (2, 40, 4), (3, 25, 3), (4, 15, 3)]


def _report(name: str, ours: np.ndarray, theirs, results: list) -> None:
    theirs = np.asarray(theirs, dtype=np.float64).reshape(-1)
    if theirs.shape != ours.shape:
        print(f"  {name:9s} SHAPE MISMATCH ours={ours.shape} theirs={theirs.shape}")
        results.append(False)
        return
    err = float(np.max(np.abs(ours - theirs)))
    ok = err < ATOL
    print(f"  {name:9s} dim={theirs.size:4d}  max|diff|={err:.3e}  {'OK' if ok else 'MISMATCH'}")
    results.append(ok)


def cross_library(results: list) -> None:
    rng = np.random.default_rng(20260726)
    for d, n, depth in CASES:
        path = np.cumsum(rng.normal(size=(n, d)), axis=0)
        ours = signature(path, depth)
        print(f"\npath d={d} n={n} depth={depth}  -> dim={ours.size}")

        try:
            import esig
            _report("esig", ours, esig.stream2sig(path, depth)[1:], results)
        except Exception as exc:
            print(f"  esig      skipped: {type(exc).__name__}: {exc}")

        try:
            import jax
            jax.config.update("jax_enable_x64", True)  # signax defaults to float32
            import jax.numpy as jnp
            import signax
            sx = signax.signature(jnp.asarray(path, dtype=jnp.float64), depth)
            flat = (np.concatenate([np.asarray(t).reshape(-1) for t in sx])
                    if isinstance(sx, (list, tuple)) else np.asarray(sx).reshape(-1))
            _report("signax", ours, flat, results)
        except Exception as exc:
            print(f"  signax    skipped: {type(exc).__name__}: {exc}")

        try:
            import roughpy as rp
            ctx = rp.get_context(width=d, depth=depth, coeffs=rp.DPReal)
            strm = rp.LieIncrementStream.from_increments(np.diff(path, axis=0), ctx=ctx)
            sig = strm.signature(rp.RealInterval(-1e6, 1e6))
            _report("roughpy", ours, np.asarray(sig)[1:], results)  # drop level 0
        except Exception as exc:
            print(f"  roughpy   skipped: {type(exc).__name__}: {exc}")


def structural(results: list) -> None:
    rng = np.random.default_rng(11)
    path = np.cumsum(rng.normal(size=(30, 2)), axis=0)
    print("\nstructural checks")

    # 1. reparameterisation invariance
    dense = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        dense += [a + (b - a) * f for f in (0.37, 0.73, 1.0)]
    err = float(np.max(np.abs(signature(path, 4) - signature(np.array(dense), 4))))
    print(f"  reparameterisation invariance  max|diff|={err:.3e}  {'OK' if err < ATOL else 'FAIL'}")
    results.append(err < ATOL)

    # 2. level 1 is the total increment
    err = float(np.max(np.abs(signature(path, 3)[:2] - (path[-1] - path[0]))))
    print(f"  level 1 == total increment     max|diff|={err:.3e}  {'OK' if err < ATOL else 'FAIL'}")
    results.append(err < ATOL)

    # 3. shuffle identity. Flat layout for d=2 is [S_1, S_2, S_11, S_12, S_21, S_22],
    #    so level 2 starts at offset 2 and entry (i,j) sits at 2 + i*d + j.
    s = signature(path, 2)
    err = float(abs((s[2 + 0 * 2 + 1] + s[2 + 1 * 2 + 0]) - s[0] * s[1]))
    print(f"  shuffle S12+S21 == S1*S2       max|diff|={err:.3e}  {'OK' if err < ATOL else 'FAIL'}")
    results.append(err < ATOL)

    # 4. a path concatenated with its reversal is tree-like
    back = np.vstack([path, path[::-1][1:]])
    err = float(np.max(np.abs(signature(back, 3))))
    print(f"  path * reverse == identity     max|S|={err:.3e}  {'OK' if err < 1e-9 else 'FAIL'}")
    results.append(err < 1e-9)


def main() -> int:
    results: list[bool] = []
    cross_library(results)
    structural(results)
    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
