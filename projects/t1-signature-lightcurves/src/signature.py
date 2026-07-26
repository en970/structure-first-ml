"""Truncated path signatures and the path augmentations used to build them.

The signature of a path X: [0, T] -> R^d is the sequence of its iterated integrals

    S(X) = ( 1, \\int dX, \\iint dX (x) dX, ... ),

with the k-th term an element of (R^d)^{otimes k}. For a piecewise-linear path the
signature has a closed form: each straight segment with increment `delta` contributes
the tensor exponential

    S(segment)_k = delta^{otimes k} / k!,

and segments combine by Chen's identity

    S(X * Y)_k = sum_{i=0}^{k} S(X)_i (x) S(Y)_{k-i}.

That is the whole algorithm. It is implemented here from scratch rather than taken from a
library for three reasons: the repository's theory track needs access to individual
tensor levels, the installation of the standard C++ backends is fragile across Python
versions, and an implementation whose correctness is verified against three independent
libraries is worth more than a dependency.

Tensor levels are stored as flat float64 arrays of length d**k in row-major
(lexicographic) index order, so entry (i_1, ..., i_k) sits at index
sum_j i_j * d**(k-j). This matches the convention used by esig and iisignature.

Verified against esig, signax and roughpy to ~1e-13 by `verify_signature.py`.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "signature",
    "signature_levels",
    "signature_dim",
    "add_time",
    "lead_lag",
    "add_basepoint",
    "cumulative_sum",
]


# --------------------------------------------------------------------- core


def signature_dim(d: int, depth: int) -> int:
    """Number of coefficients in a depth-truncated signature, excluding level 0."""
    return sum(d ** k for k in range(1, depth + 1))


def _exp_increment(delta: np.ndarray, depth: int) -> list[np.ndarray]:
    """Signature levels of a single straight segment: delta^{otimes k} / k!."""
    levels: list[np.ndarray] = []
    cur = np.array([1.0])
    fact = 1.0
    for k in range(1, depth + 1):
        cur = np.outer(cur, delta).reshape(-1)
        fact *= k
        levels.append(cur / fact)
    return levels


def _chen(a: list[np.ndarray], b: list[np.ndarray], depth: int) -> list[np.ndarray]:
    """Chen's identity: combine the signatures of two concatenated paths.

    Level 0 is identically 1 and is not stored, which is why the i=0 and i=k terms
    reduce to `a[k-1]` and `b[k-1]`.
    """
    out: list[np.ndarray] = []
    for k in range(1, depth + 1):
        acc = a[k - 1] + b[k - 1]
        for i in range(1, k):
            acc = acc + np.outer(a[i - 1], b[k - i - 1]).reshape(-1)
        out.append(acc)
    return out


def signature_levels(path: np.ndarray, depth: int) -> list[np.ndarray]:
    """Truncated signature of a piecewise-linear path, as a list of tensor levels.

    Parameters
    ----------
    path : array of shape (n_points, d)
        Vertices of the piecewise-linear path, in order.
    depth : int
        Truncation depth m; levels 1 through m are returned.

    Returns
    -------
    list of arrays, the k-th having length d**(k+1).
    """
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2:
        raise ValueError(f"path must be 2-D (n_points, d), got shape {path.shape}")
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")

    d = path.shape[1]
    if path.shape[0] < 2:
        # A single point has the trivial signature.
        return [np.zeros(d ** k) for k in range(1, depth + 1)]

    acc: list[np.ndarray] | None = None
    for delta in np.diff(path, axis=0):
        seg = _exp_increment(delta, depth)
        acc = seg if acc is None else _chen(acc, seg, depth)
    assert acc is not None
    return acc


def signature(path: np.ndarray, depth: int) -> np.ndarray:
    """Truncated signature as a single flat feature vector (level 0 omitted)."""
    return np.concatenate(signature_levels(path, depth))


# ------------------------------------------------------------ augmentations
#
# The signature is invariant under reparameterisation, which is sometimes exactly what
# is wanted and sometimes destroys information that matters. These transforms control
# that, and which one is appropriate is an empirical question, not a default.


def add_time(path: np.ndarray, times: np.ndarray | None = None) -> np.ndarray:
    """Prepend a time channel, breaking reparameterisation invariance deliberately.

    Without this the signature cannot distinguish a fast event from a slow one of the
    same shape. For variable stars, where period is physical, the time channel is
    usually necessary; for shape-driven classification it may be harmful. Both cases are
    tested rather than assumed.
    """
    path = np.asarray(path, dtype=np.float64)
    n = path.shape[0]
    t = np.linspace(0.0, 1.0, n) if times is None else np.asarray(times, dtype=np.float64)
    if t.shape[0] != n:
        raise ValueError(f"times has length {t.shape[0]}, path has {n} points")
    return np.column_stack([t, path])


def lead_lag(path: np.ndarray) -> np.ndarray:
    """Lead-lag transform: pair each channel with a delayed copy of itself.

    This is what makes quadratic variation visible to the signature. For a path with
    d channels the output has 2d channels and 2n-1 points; the level-2 terms of the
    resulting signature then carry the realised variance and covariance of the original
    path, which a plain signature of a piecewise-linear interpolant cannot see.
    """
    path = np.asarray(path, dtype=np.float64)
    n, d = path.shape
    out = np.empty((2 * n - 1, 2 * d), dtype=np.float64)
    for i in range(2 * n - 1):
        lead = (i + 1) // 2
        lag = i // 2
        out[i, :d] = path[lead]
        out[i, d:] = path[lag]
    return out


def add_basepoint(path: np.ndarray) -> np.ndarray:
    """Prepend the origin, making the signature sensitive to absolute position.

    The signature depends only on increments, so two paths differing by a translation
    have identical signatures. Where the absolute level matters -- a star's mean
    magnitude, for instance -- the basepoint restores it.
    """
    path = np.asarray(path, dtype=np.float64)
    return np.vstack([np.zeros((1, path.shape[1])), path])


def cumulative_sum(path: np.ndarray) -> np.ndarray:
    """Integrate the path, turning a sequence of observations into a cumulative stream.

    Useful when the natural data object is a set of increments rather than a level.
    """
    return np.cumsum(np.asarray(path, dtype=np.float64), axis=0)
