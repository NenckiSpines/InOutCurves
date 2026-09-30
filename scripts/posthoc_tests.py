"""Pairwise follow-up with Holm family-wise multiplicity correction.

The uploaded project supplied a two-group statistic/test, not a complete post-hoc
family. This module reuses that two-group engine for ALL selected group pairs.
Each pair has its OWN pooled subjects and null distribution. Do not use the
three-group complete-null permutations to test a single pair when another group
can differ. Holm correction covers the complete, prespecified pair family.

All pair p-values are reported for transparency. By default a confirmatory
pairwise rejection also requires a significant omnibus test. The ungated Holm
decision is exported separately; the gate never changes the adjusted p-values.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
from .permutation_tests import permutation_test


def holm_adjust(pvalues) -> np.ndarray:
    p = np.asarray(pvalues,dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Holm correction needs a nonempty finite p-value vector in [0,1].')
    order = np.argsort(p,kind='stable')
    corrected = np.minimum(1.0,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    out = np.empty_like(p)
    out[order] = corrected
    return out


def pairwise_posthoc(groups, x, *, labels=None, permutations: int = 9999,
                     seed: int = 20260919, alpha: float = 0.05,
                     omnibus_pvalue: float | None = None,
                     require_omnibus: bool = True, exact: str = 'auto',
                     exact_limit: int = 50000) -> list[dict]:
    if len(groups) < 3:
        raise ValueError('Post-hoc family requires at least three independent groups.')
    if not 0 < alpha < 1 or seed < 0:
        raise ValueError('Require 0 < alpha < 1 and a nonnegative seed.')
    if require_omnibus and (omnibus_pvalue is None or not 0 < omnibus_pvalue <= 1):
        raise ValueError('A valid omnibus p-value is required for gated post-hoc decisions.')
    labels = list(labels) if labels is not None else [f'Group {i+1}' for i in range(len(groups))]
    if len(labels) != len(groups) or len(set(labels)) != len(labels):
        raise ValueError('One unique label per group is required.')
    rows=[]
    for index,(i,j) in enumerate(combinations(range(len(groups)),2)):
        result = permutation_test((groups[i],groups[j]),x,permutations=permutations,
                                  rng=np.random.default_rng(np.random.SeedSequence([seed,301,index])),
                                  exact=exact,exact_limit=exact_limit)
        rows.append({'group_1':labels[i],'group_2':labels[j],
                     'n_1':len(groups[i]),'n_2':len(groups[j]),
                     'l2_squared':result.statistic,'p_raw':result.pvalue,
                     'method':result.method,'allocations_evaluated':result.evaluated_permutations})
    adjusted = holm_adjust([r['p_raw'] for r in rows])
    gate = not require_omnibus or omnibus_pvalue <= alpha
    for row,p in zip(rows,adjusted):
        row.update({'p_holm':float(p),'reject_holm':bool(p<=alpha),
                    'omnibus_gate_passed':bool(gate),'reject_after_gate':bool(gate and p<=alpha),
                    'alpha':alpha,'family_size':len(rows),'adjustment':'Holm'})
    return rows
