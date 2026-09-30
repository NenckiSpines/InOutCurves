"""Optional exploratory sigmoid fit. NOT used by the nonparametric test.

Retains the earlier negative-response logistic convention, but removes global
variables, duplicate sign inversions, import-time plotting and unused packages.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit


@dataclass
class SigmoidFitter:
    xdata: np.ndarray
    ydata: np.ndarray

    @staticmethod
    def sigmoid(x,L,x0,k,b):
        return -(L*expit(k*(np.asarray(x)-x0))+b)

    def fit_sigmoid(self,p0=None):
        x,y=np.asarray(self.xdata,dtype=float),np.asarray(self.ydata,dtype=float)
        if x.ndim!=1 or y.shape!=x.shape or len(x)<4 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Sigmoid fitting needs matching finite x/y vectors with at least four points.')
        self.parameters,self.covariance=curve_fit(self.sigmoid,x,y,p0=p0,method='dogbox',maxfev=100000)
        self.y_fitted=self.sigmoid(x,*self.parameters)
        return self.parameters

    def r_squared(self):
        if not hasattr(self,'y_fitted'):raise ValueError('Fit the model before requesting R squared.')
        y=np.asarray(self.ydata,dtype=float)
        total=float(np.sum((y-y.mean())**2))
        return float('nan') if total==0 else 1-float(np.sum((y-self.y_fitted)**2))/total
