"""Exact squared-L2 integrals for piecewise-linear curves on a common grid.

l2_squared returns D_ij = ||mean_i - mean_j||_L2**2, with no square root.
l2_grand returns SUM(D_ij), not SUM(D_ij**2): do not square twice.
For three groups this is L2_Grand = L_12**2 + L_23**2 + L_13**2.
"""
from itertools import combinations
import numpy as np


def validate_grid(x) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError('The common stimulus grid must be finite and strictly increasing.')
    return x


def l2_squared(a, b, x) -> float:
    x = validate_grid(x)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != x.shape or b.shape != x.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Both curves must be finite and match the common grid.')
    d = a-b
    return float(np.sum(np.diff(x)*(d[:-1]**2+d[:-1]*d[1:]+d[1:]**2)/3.0))


def integration_cholesky(x) -> np.ndarray:
    x = validate_grid(x)
    dx = np.diff(x)
    q = np.zeros((len(x), len(x)))
    i = np.arange(len(dx))
    q[i, i] += dx/3
    q[i+1, i+1] += dx/3
    q[i, i+1] = q[i+1, i] = dx/6
    return np.linalg.cholesky(q)


def l2_grand(means, x) -> float:
    means = np.asarray(means, dtype=float)
    if means.ndim != 2 or len(means) < 2:
        raise ValueError('Supply at least two group-mean curves.')
    return sum(l2_squared(means[i], means[j], x) for i, j in combinations(range(len(means)), 2))
