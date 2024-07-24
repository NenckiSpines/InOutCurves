
from utilities import get_p_value, draw_l2_histogram
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import mannwhitneyu
import scipy.stats
import pingouin as pg
import pandas as pd
import os
# Importing libraries 
import statsmodels.api as sm 
from statsmodels.formula.api import ols 
from statsmodels.stats.anova import AnovaRM 

class SigmoidFitter:
    def __init__(self, xdata, ydata):
        self.xdata = xdata
        self.ydata = ydata

    def sigmoid(self, x, L, x0, k, b):
        y = L / (1 + np.exp(-k*(x-x0))) + b
        return -y

    def fit_sigmoid(self, p0=None):
        popt, pcov = curve_fit(self.sigmoid, self.xdata, self.ydata, p0=p0, method='dogbox', maxfev=100000)
        plotted_curve = lambda x: -sigmoid_fitter.sigmoid(x, *popt)
        self.y_fitted = np.array(list(map(plotted_curve, xdata)))
        print ("fit params",popt)
        return popt

    def r_squared(self, y_data,y_fitted):
        y_mean = np.mean(y_data)
        ss_total = np.sum((y_data - y_mean) ** 2)
        ss_residual = np.sum((y_data - y_fitted) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        return r_squared

    def print_result(self):
        print("R-squared:", self.r_squared(-self.ydata,self.y_fitted))


    if __name__ == "__main__":
        xdata=[0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0, 275.0, 300.0]	
        ydata=[]	
        p0 = [1.0871044, 5.00, 0.0108271, -1.0871044/2]
        p0 = [ 1.47079148e-01 , 1.68051020e+02, -5.00489901e+00 ,-1.32150433e-01]
        p0 = [ 1 , 140, 0.1 ,0]
        sigmoid_fitter = SigmoidFitter(xdata, ydata)
        sg= sigmoid_fitter.sigmoid(np.array(xdata),*p0)
        fig1=plt.figure(dpi=60)
        ax1=fig1.add_subplot(111)
        ax1.plot(sg)
        plt.show()