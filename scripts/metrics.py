"""Rejection rates with binomial uncertainty; no ambiguous 'accuracy' labels."""
import math
import numpy as np
from scipy import stats


def binomial_interval(successes: int,trials: int) -> tuple[float,float]:
    if not isinstance(trials,(int,np.integer)) or trials<1 or not 0<=successes<=trials:
        raise ValueError('Need 0 <= successes <= a positive trial count.')
    low=0.0 if successes==0 else float(stats.beta.ppf(0.025,successes,trials-successes+1))
    high=1.0 if successes==trials else float(stats.beta.ppf(0.975,successes+1,trials-successes))
    return low,high


def rejection_rate(pvalues,alpha: float=0.05) -> dict:
    p=np.asarray(pvalues,dtype=float)
    if p.ndim!=1 or not p.size or not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not 0<alpha<1:
        raise ValueError('Supply finite p-values in [0,1] and 0 < alpha < 1.')
    hits=int(np.count_nonzero(p<=alpha));n=len(p);q=hits/n
    lo,hi=binomial_interval(hits,n)
    return {'alpha':alpha,'experiments':n,'rejections':hits,'estimate':q,'ci_low':lo,'ci_high':hi,
            'mcse':math.sqrt(q*(1-q)/n)}


class RateCalculator:
    """Retained role of the previous calculator, with unambiguous output names."""
    def calculate_rates(self,null,alternative,alphas=(0.05,0.01,0.001)) -> dict:
        self.results={}
        for alpha in alphas:
            fp,power=rejection_rate(null,alpha),rejection_rate(alternative,alpha)
            self.results[alpha]={'false_positive_rate':fp,'power':power,
                                 'false_negative_rate':1-power['estimate'],
                                 'specificity':1-fp['estimate'],
                                 'balanced_accuracy':(1-fp['estimate']+power['estimate'])/2}
        return self.results
