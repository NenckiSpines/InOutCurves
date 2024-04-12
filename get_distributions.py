from animal_results import AnimalResults
from utilities import get_p_value, draw_l2_histogram
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import mannwhitneyu
import scipy.stats
import pingouin as pg
import pandas as pd
# Importing libraries 
import statsmodels.api as sm 
from statsmodels.formula.api import ols 
from statsmodels.stats.anova import AnovaRM 
from ssqError import DistributionFitter
import numpy
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'lime', 'teal', 'navy', 'maroon', 'gold', 'indigo', 'darkred']

def find_minimal_a(x, y):
    # Stack the x vector into a matrix with an additional column of ones
    X = np.vstack((x, np.ones(len(x)))).T
    
    # Use least squares to find the optimal 'a' parameter
    a, _ = np.linalg.lstsq(X, y, rcond=None)[0]
    
    return a

# Modify data
def modfunc(xv, yv):
    a = 0.02
    omega = 2 * np.pi / 300.0
    res = []
    for x, y in zip(xv, yv):
        yn = y + a * np.sin(omega * x)
        res.append(yn)
    return np.array(res)


class SigmoidFitter:
    def __init__(self, xdata, ydata):
        self.xdata = xdata
        self.ydata = ydata

    def sigmoid(self, x, L, x0, k, b):
        y = L / (1 + np.exp(-k*(x-x0))) + b
        return y

    def fit_sigmoid(self, p0=None):
        popt, pcov = curve_fit(self.sigmoid, self.xdata, self.ydata, p0=p0, method='dogbox', maxfev=10000)
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
   
         

# Load data
animal_results = AnimalResults.from_csv("LTP5_slope.csv")
#animal_results = AnimalResults.from_csv("result.csv")
grouped_by_animal = animal_results.group_by_animal()
#p2=grouped_by_animal.plot("Grouped by animal")
#print (p2)
xdata = grouped_by_animal.currents
grouped_by_group = grouped_by_animal.group_by_group()
#p3=grouped_by_group.plot("Grouped by group")
l2 = grouped_by_group.get_l2()
randomized_group_l2s = [grouped_by_animal.randomize_groups().group_by_group().get_l2() for _ in range(1000)]
figure, axes = plt.subplots()
draw_l2_histogram(axes=axes, results=randomized_group_l2s, threshold=l2, bin_count=100)
print ("Randomization p-value=",get_p_value(randomized_group_l2s, l2))

fig1=plt.figure(dpi=60)
ax1=fig1.add_subplot(111)
grouped_by_animal.plot(ax1)

fig1a=plt.figure(dpi=60)
ax1a=fig1a.add_subplot(111)
grouped_by_group.plot(ax1a)

avg_res = grouped_by_animal.get_average_responses()


p0 = [1.0871044, 0.00, 0.0108271, -1.0871044/2]


fig5=plt.figure(dpi=60)
ax5=fig5.add_subplot(111)
res={}
print (len(grouped_by_animal.results))
pre_data={}
post_data={}
coeffs=[]
residuals=[]
for item in grouped_by_animal.results:
	if item.group_id=="PRE":
		pre_data[item.animal_id]=item.responses
	else:
		post_data[item.animal_id]=item.responses
	acoeff=find_minimal_a(avg_res, item.responses)
	model_response=numpy.multiply(avg_res, acoeff)
	residuals.append(numpy.subtract(model_response,item.responses))
	coeffs.append(acoeff)
	
residuals=numpy.array(residuals)
variance=numpy.var(residuals,axis=0)
print ("variance=",variance)
std=numpy.sqrt(variance)
print ("std=",std)
	
fig2=plt.figure(dpi=60)
ax2=fig2.add_subplot(111)
ax2.plot(std)		
	
for id in list(post_data.keys()):	
	diff=numpy.subtract(post_data[id],pre_data[id])
	#ax1.plot(post_data[id],color=colors[int(id)%19],marker=".")
	#ax1.plot(post_data[id],color=colors[int(id)%19],marker=".")
	ax5.plot(diff,color=colors[int(id)%19])






# Create an instance of DistributionFitter
fitter = DistributionFitter(coeffs)

# Fit using Anderson-Darling test
best_distribution_ad, best_params_ad, ad_stat = fitter.fit_best_distribution(method='ad')

# Fit using sum of squares error
best_distribution_sse, best_params_sse, sse_stat = fitter.fit_best_distribution(method='sse')

print("Fitting using Anderson-Darling Test:")
print("Best Distribution:", best_distribution_ad.name)
print("Best Parameters:", best_params_ad)
print("Anderson-Darling Test Statistic:", ad_stat)
print()

print("Fitting using Least Squares Error:")
print("Best Distribution:", best_distribution_sse.name)
print("Best Parameters:", best_params_sse)
print("Sum of Squares Error:", sse_stat)


# Plotting both distributions
fig3=plt.figure(dpi=60,figsize=(12, 5))
ax31=fig3.add_subplot(211)
ax32=fig3.add_subplot(212)
# Plotting for Anderson-Darling

fitter.plot_fitted_distribution(ax31,best_distribution_ad, best_params_ad)
fitter.plot_fitted_distribution(ax32,best_distribution_sse, best_params_sse)


plt.show()