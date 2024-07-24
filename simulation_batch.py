import os

import matplotlib.pyplot as plt
import numpy as np
from numpy import geomspace

from animal_results import AnimalResults
from constants import PLOT_COLORS
from rate_calc import RateCalculator
from reference_tests import RefTests
from utilities import get_p_value

DYNAMIC=0

class Plotter:
    @staticmethod
    def calculate_optimal_bins(data):
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        h = 2 * iqr / (len(data) ** (1/3))
        optimal_bins = int(np.ceil((np.log10(max(data)) - np.log10(min(data))) / h))
        return optimal_bins

    @staticmethod
    def draw_histogram(data,ax=None,color="blue"):
        if ax==None:    
            fig, ax = plt.subplots()
        data_clipped = np.clip(data, a_min=0.0001, a_max=None)
        #optimal_bins_nr = Plotter.calculate_optimal_bins(data_clipped)
        optimal_bins_nr =100
        bins = np.logspace(np.log10(0.0001), np.log10(1.0), optimal_bins_nr)
        hist, bin_edges = np.histogram(data_clipped, bins=bins)
        max_count = max(hist)

        if np.any(bin_edges <= 0):
            raise ValueError("Bin edges must be positive.")

        for i, count in enumerate(hist):
            bin_width = bin_edges[i + 1] - bin_edges[i]
            x = np.log10(bin_edges[i])
            y = 0
            width = np.log10(bin_edges[i + 1]) - np.log10(bin_edges[i])
            height = count
            ax.bar(x, height, width=width, bottom=y, align='edge', color=color, edgecolor=None,alpha=0.2)

        tick_positions = np.log10([0.1, 0.05, 0.01, 0.001])
        tick_labels = ['0.1', '0.05', '0.01', '0.001']
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram with Log Scale on X-axis')

        ax.invert_xaxis()
    @staticmethod
    def draw_histograms(p_values):
        fig, ax = plt.subplots()
        Plotter.draw_histogram(p_values[:,0],ax)
        Plotter.draw_histogram(p_values[:,1],ax,color="orange")

    @staticmethod
    def draw_histogram_2(p_values):
        fig, ax = plt.subplots()
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("Actual values")
        ax.set_ylabel("Frequency")
        ax.hist(
            x=p_values[:, 0],
            bins=geomspace(start=0.0001, stop=1, num=100),
            color=PLOT_COLORS[2],
            fill=True,
            histtype="step",
        )
        ax.hist(
            x=p_values[:, 1],
            bins=geomspace(start=0.0001, stop=1, num=100),
            color=PLOT_COLORS[1],
            fill=False,
            histtype="step",
        )
    
    @staticmethod    
    def plot_p_values(p_values,color="blue"):
        fig10=plt.figure(dpi=60)
        ax10=fig10.add_subplot(111)
        ax10.plot(p_values[:,0],p_values[:,1],ls="None",marker=".",color=color)
        pvm=np.max(p_values[:,0])
        sl=np.arange(0,pvm,pvm/10)
        ax10.plot(sl,sl,ls="-",color="black",marker=None)

class Analyzer:
    quiet: bool

    def __init__(self, quiet=False):
        self.p_values = []
        self.quiet = quiet

    def analyze_data(self, animal_results, ref_tests, iterations=100, keep_differences=False, animals_per_group=10):
        self.p_values = []
        for i in range(iterations):
            if not self.quiet:
                print(i)
            res = []
            fake = animal_results.to_fake(keep_differences=keep_differences, animals_per_group=animals_per_group)
            grouped_by_animal = fake.group_by_animal()
            ref_tests.makePdFrame(grouped_by_animal)
            p_value1 = ref_tests.p_anova()
            grouped_by_group = grouped_by_animal.group_by_group()
            l2 = grouped_by_group.get_l2()
            randomized_group_l2s=[]
            if DYNAMIC: 
                count=0
                while ((count<100)&(len(randomized_group_l2s)<100000)):
                    randomized_group_l2s = randomized_group_l2s+[grouped_by_animal.randomize_groups().group_by_group().get_l2() for _ in range(10000)]
                    arr = np.array(randomized_group_l2s)
                    count = np.sum(arr > l2)
                    if not self.quiet:
                        print(l2, count, len((randomized_group_l2s)))
            else:
                randomized_group_l2s = randomized_group_l2s+[grouped_by_animal.randomize_groups().group_by_group().get_l2() for _ in range(10000)]
            p_value = get_p_value(randomized_group_l2s, l2)
            if not self.quiet:
                print("p_values=", p_value, p_value1)
            res.append(p_value)
            res.append(p_value1)
            self.p_values.append(res)
        self.p_values = np.array(self.p_values)

    def plot_p_values(self,color="blue"):
        fig10=plt.figure(dpi=60)
        ax10=fig10.add_subplot(111)
        ax10.plot(self.p_values[:,0],self.p_values[:,1],ls="None",marker=".",color=color)
      

    def calculate_mean(self):
        pass
        #print(np.mean(self.p_values),axis=1)

if __name__ == "__main__":

    SAME_SUBJECTS = 0


    animal_results = AnimalResults.from_csv(path="LTP5_slope_diffsubjects.csv")
    file_no_diff = "data/p_values_NO_17.csv"
    file_diff = "data/p_values_DIFF_17.csv"


    ref_tests = RefTests(SAME_SUBJECTS)

    if not (os.path.exists(file_no_diff)):
        analyzer = Analyzer()
        analyzer.analyze_data(animal_results, ref_tests)
        analyzer.plot_p_values()
        analyzer.calculate_mean()
        data_NO=analyzer.p_values
        np.savetxt(file_no_diff,analyzer.p_values, delimiter=',')
    else:
        data_NO = np.genfromtxt(file_no_diff, delimiter=',')
    if not (os.path.exists(file_diff)):
        analyzer = Analyzer()
        analyzer.analyze_data(animal_results, ref_tests,keep_differences=True)
        analyzer.plot_p_values(color="red")
        analyzer.calculate_mean()
        data_WD=analyzer.p_values
        np.savetxt(file_diff,analyzer.p_values, delimiter=',')
    else:
        data_WD = np.genfromtxt(file_diff, delimiter=',')
        

    Plotter.draw_histogram_2(data_WD)
    Plotter.draw_histogram_2(data_NO)
    Plotter.plot_p_values(data_NO)
    Plotter.plot_p_values(data_WD,color="red")

    rc = RateCalculator()
    print ("METHOD 1")
    rc.calculate_rates(data_NO[:,0], data_WD[:,0] )
    rc.print_rates()
    print ("METHOD 2")
    rc.calculate_rates(data_NO[:,1], data_WD[:,1] )
    rc.print_rates()

    plt.show()