"""Conventional parametric comparators with explicit effect definitions.

Independent groups: mixed-design ANOVA Group main effect, equivalent to a one-way
F test on SUBJECT arithmetic means over the common, complete stimulation grid.
Paired conditions: Condition main effect, equivalent to the squared paired t
statistic on these means. Neither is a condition/group-by-stimulation interaction
or a joint test of every possible mean-function difference. No stimulus point is
counted as an independent subject. Parametric model assumptions still apply.
"""
import math
import numpy as np
from scipy import stats


def anova_group_effect(*groups) -> tuple[float,float]:
    if len(groups)==1:
        groups=tuple(groups[0])
    if len(groups)<2:
        raise ValueError('ANOVA needs at least two groups.')
    arrays=[np.asarray(a,dtype=float) for a in groups]
    if any(a.ndim!=2 or len(a)<2 or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Finite subject-by-stimulus arrays with at least two subjects are required.')
    if len({a.shape[1] for a in arrays})!=1:
        raise ValueError('ANOVA assumes a common complete stimulus grid.')
    samples=[a.mean(1) for a in arrays]
    n=sum(map(len,samples));k=len(samples)
    grand=sum(v.sum() for v in samples)/n
    between=sum(len(v)*(v.mean()-grand)**2 for v in samples)
    within=sum(np.sum((v-v.mean())**2) for v in samples)
    if within==0:
        return (0.0,1.0) if between==0 else (math.inf,0.0)
    f=float((between/(k-1))/(within/(n-k)))
    return f,float(stats.f.sf(f,k-1,n-k))


def paired_condition_effect(a,b) -> tuple[float,float]:
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.ndim!=2 or a.shape!=b.shape or len(a)<2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Paired comparator requires aligned finite subject-by-stimulus arrays.')
    d=a.mean(1)-b.mean(1)
    variance=float(np.var(d,ddof=1))
    mean=float(d.mean())
    if variance==0:
        return (0.0,1.0) if mean==0 else (math.inf,0.0)
    f=float(len(d)*mean**2/variance)
    return f,float(stats.f.sf(f,1,len(d)-1))
