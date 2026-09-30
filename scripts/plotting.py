"""Shared poster styling; kept consistent with the supplied blue/orange/green plots."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cycler import cycler

COLORS=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']


def configure_plots(font_scale: float=1.0) -> None:
    plt.rcdefaults()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11*font_scale,
        'axes.labelsize':12*font_scale,'axes.titlesize':13*font_scale,
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
        'legend.fontsize':9*font_scale,'xtick.labelsize':10*font_scale,'ytick.labelsize':10*font_scale,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.prop_cycle':cycler(color=COLORS)})
