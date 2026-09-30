"""Subject-level two-/multi-group inference and two-condition paired inference.

Independent: complete curves are reassigned, preserving every group size.
Paired: complete PRE/POST curves are swapped within each matched subject;
this is equivalent to independently sign-flipping subject difference curves.

The independent null requires exchangeability of subject curves across groups;
the paired null requires within-subject condition-label exchangeability (or
symmetric difference curves for a sign-flip interpretation), not just zero means.
Stimulus points are never shuffled. No automatic heteroscedasticity correction.

Exact tail = inclusive exceedances / all distinct allocations.
Monte Carlo tail = (1 + inclusive exceedances)/(B + 1).
The distance statistic is nondirectional: use the upper tail, not a doubled tail.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import math
from typing import Sequence
import numpy as np
from .distances import integration_cholesky

EXACT_LIMIT = 50000
BATCH_SIZE = 512


@dataclass
class PermutationResult:
    statistic: float
    pvalue: float
    design: str
    method: str
    requested_permutations: int
    evaluated_permutations: int
    total_allocations: int
    group_sizes: tuple[int, ...]
    null_distribution: np.ndarray | None = None

    def as_dict(self, include_distribution: bool = False) -> dict:
        value = {k: v for k, v in vars(self).items() if k != 'null_distribution'}
        if include_distribution and self.null_distribution is not None:
            value['null_distribution'] = self.null_distribution.tolist()
        return value


def allocation_count(sizes: Sequence[int]) -> int:
    if len(sizes) < 2 or any(int(n) != n or n < 1 for n in sizes):
        raise ValueError('At least two positive integer group sizes are required.')
    remaining, total = sum(sizes), 1
    for n in sizes[:-1]:
        total *= math.comb(remaining, int(n))
        remaining -= int(n)
    return total


@lru_cache(maxsize=8)
def partition_indices(sizes: tuple[int, ...]) -> np.ndarray:
    """One canonical within-group ordering per allocation of labelled groups."""
    if allocation_count(sizes) > EXACT_LIMIT:
        raise ValueError(f'Enumeration exceeds the safety limit of {EXACT_LIMIT}.')

    def recurse(available: tuple[int, ...], remaining: tuple[int, ...]):
        if len(remaining) == 1:
            yield available
            return
        for first in combinations(available, remaining[0]):
            selected = set(first)
            other = tuple(i for i in available if i not in selected)
            for rest in recurse(other, remaining[1:]):
                yield first+rest
    return np.asarray(list(recurse(tuple(range(sum(sizes))), sizes)), dtype=np.intp)


def tail_threshold(observed: float) -> float:
    return observed - 100*np.finfo(float).eps*abs(observed)


def inclusive_pvalue(values, observed: float, exact: bool = False) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not np.isfinite(observed):
        raise ValueError('Finite observed and nonempty null statistics are required.')
    count = int(np.count_nonzero(values >= tail_threshold(observed)))
    p = count/len(values) if exact else (count+1)/(len(values)+1)
    if not 0 < p <= 1:
        raise ArithmeticError('Exact distribution lacks the observed allocation or numerical ties were lost.')
    return float(p)


def _validate_groups(groups, chol) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(a, dtype=float) for a in groups)
    if len(arrays) < 2 or chol.ndim != 2 or chol.shape[0] != chol.shape[1]:
        raise ValueError('At least two groups and a square integration factor are required.')
    for a in arrays:
        if a.ndim != 2 or len(a) < 2 or a.shape[1] != len(chol) or not np.isfinite(a).all():
            raise ValueError('Each group must have at least two finite subject curves on the same grid.')
    if not np.isfinite(chol).all():
        raise ValueError('Nonfinite integration factor.')
    return arrays


def independent_distribution(groups, chol: np.ndarray, draws: int, rng: np.random.Generator,
                             exact: bool = False, batch_size: int = BATCH_SIZE) -> tuple[float, np.ndarray]:
    arrays = _validate_groups(groups, chol)
    if draws < 1 or batch_size < 1:
        raise ValueError('draws and batch_size must be positive.')
    sizes = tuple(len(a) for a in arrays)
    pooled = np.concatenate(arrays)
    z = (pooled-pooled.mean(0)) @ chol
    cuts = np.cumsum((0,)+sizes)
    means = np.stack([z[cuts[i]:cuts[i+1]].mean(0) for i in range(len(sizes))])
    pairs = tuple(combinations(range(len(sizes)), 2))
    observed = float(sum(np.sum((means[i]-means[j])**2) for i,j in pairs))
    choices = partition_indices(sizes) if exact else None
    count = len(choices) if exact else draws
    output = np.empty(count)
    total_z = z.sum(0)
    for start in range(0, count, batch_size):
        end = min(count, start+batch_size)
        if len(sizes) == 2:
            # Same uniform-subset algorithm as the uploaded poster workflows.
            take = (choices[start:end, :sizes[0]] if exact else
                    np.argpartition(rng.random((end-start, len(z))), sizes[0]-1, axis=1)[:, :sizes[0]])
            first = z[take].sum(1)
            d = first/sizes[0]-(total_z-first)/sizes[1]
            output[start:end] = np.einsum('ij,ij->i', d, d)
        else:
            # A uniform permutation partitioned into fixed-size labelled groups.
            order = choices[start:end] if exact else np.argsort(rng.random((end-start, len(z))), axis=1)
            group_means = [z[order[:, cuts[g]:cuts[g+1]]].mean(1) for g in range(len(sizes))]
            values = np.zeros(end-start)
            for i,j in pairs:
                d = group_means[i]-group_means[j]
                values += np.einsum('ij,ij->i', d, d)
            output[start:end] = values
    return observed, output


def paired_distribution(a, b, chol: np.ndarray, draws: int, rng: np.random.Generator,
                        exact: bool = False, batch_size: int = BATCH_SIZE) -> tuple[float, np.ndarray]:
    a, b = _validate_groups((a,b), chol)
    if a.shape != b.shape:
        raise ValueError('Paired arrays must have the same shape and be aligned by subject ID.')
    if draws < 1 or batch_size < 1:
        raise ValueError('draws and batch_size must be positive.')
    n = len(a)
    if exact and 2**n > EXACT_LIMIT:
        raise ValueError('Paired exact enumeration exceeds the safety limit.')
    differences = (a-b) @ chol
    observed = float(np.sum(differences.mean(0)**2))
    count = 2**n if exact else draws
    output = np.empty(count)
    for start in range(0, count, batch_size):
        end = min(count, start+batch_size)
        if exact:
            bits = (np.arange(start,end,dtype=np.uint64)[:,None] >> np.arange(n,dtype=np.uint64)) & 1
            signs = 1.0-2.0*bits
        else:
            signs = 1.0-2.0*rng.integers(0,2,size=(end-start,n))
        delta = signs @ differences / n
        output[start:end] = np.einsum('ij,ij->i', delta,delta)
    return observed, output


def test_with_cholesky(groups, chol, *, design: str = 'independent', permutations: int = 9999,
                       rng: np.random.Generator | None = None, exact: str = 'auto',
                       exact_limit: int = EXACT_LIMIT, return_distribution: bool = False) -> PermutationResult:
    arrays = _validate_groups(groups, np.asarray(chol))
    if not isinstance(permutations, (int,np.integer)) or permutations < 1 or exact_limit < 0:
        raise ValueError('A positive integer permutation count and nonnegative exact limit are required.')
    if exact not in ('auto','always','never'):
        raise ValueError('exact must be auto, always, or never.')
    if design == 'paired':
        if len(arrays) != 2 or arrays[0].shape != arrays[1].shape:
            raise ValueError('Paired inference requires exactly two aligned, equal-sized groups.')
        total = 2**len(arrays[0])
    elif design == 'independent':
        total = allocation_count(tuple(map(len, arrays)))
    else:
        raise ValueError('Unknown design; choose paired or independent.')
    ceiling = min(exact_limit, EXACT_LIMIT)
    if exact == 'always' and total > ceiling:
        raise ValueError(f'Exact enumeration needs {total:,} allocations; safety ceiling is {ceiling:,}.')
    use_exact = exact == 'always' or (exact == 'auto' and total <= min(permutations, ceiling))
    rng = np.random.default_rng() if rng is None else rng
    if design == 'paired':
        observed, values = paired_distribution(*arrays, chol, permutations, rng, use_exact)
    else:
        observed, values = independent_distribution(arrays, chol, permutations, rng, use_exact)
    p = inclusive_pvalue(values, observed, use_exact)
    return PermutationResult(observed,p,design,'exact' if use_exact else 'monte_carlo',
                             int(permutations),len(values),total,tuple(map(len,arrays)),
                             values if return_distribution else None)


def permutation_test(groups, x, **kwargs) -> PermutationResult:
    """Array API: callers of paired tests must align rows; dataset API validates IDs."""
    return test_with_cholesky(groups, integration_cholesky(x), **kwargs)
