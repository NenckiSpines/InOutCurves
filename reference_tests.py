from statsmodels.stats.anova import AnovaRM 
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import mannwhitneyu
import scipy.stats
import pingouin as pg
import pandas as pd

class RefTests():
	def __init__(self,SAME_SUBJECTS):
		dataframe=None
		self.SAME_SUBJECTS=SAME_SUBJECTS
	def makePdFrame(self,grouped_by_animal):
		xdata = grouped_by_animal.currents
		dataGroup=[]
		dataSubject=[]
		dataResponse=[]
		dataInput=[]

		for item in grouped_by_animal.results:
			dataResponse=dataResponse+list(item.responses)
			dataInput=dataInput+list(xdata)
			for _ in range(len(xdata)):
				dataGroup.append(item.group_id)
				anid=item.animal_id
				#if item.animal_id>18:
				#	anid=item.animal_id-18
				dataSubject.append(anid)
		self.dataframe = pd.DataFrame({'Voltage': dataInput, 
									  'Group':dataGroup, 
									  'Output': dataResponse,"Subject":dataSubject})
	def p_anova(self):
		if not self.SAME_SUBJECTS:
			aov = pg.mixed_anova(dv='Output', within='Voltage', between='Group', subject='Subject', data=self.dataframe)	
			print (aov)
			print ("Mixed model anova pvalue=",(aov['p-unc'][0]))
			return(aov['p-unc'][0])
		if self.SAME_SUBJECTS:
			aov=AnovaRM(data=self.dataframe, depvar='Output', subject='Subject', within=['Group','Voltage']).fit()
			print (aov)
			return(aov)