from itertools import tee

from matplotlib.pyplot import subplots
from matplotlib.transforms import ScaledTranslation
from numpy import geomspace

from constants import PLOT_LINE_WIDTH, PLOT_COLORS, P_VALUE_LABEL, OCCURRENCE_COUNT_LABEL


def get_grid_figure(row_count=1, column_count=1, share_y=False):
    figure, axes = subplots(nrows=row_count, ncols=column_count, constrained_layout=True, dpi=300, sharey=share_y)
    figure.set_figwidth(6.4)
    for a in axes.flat:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.set_box_aspect(0.75)
    return figure, axes


def get_p_value(results, threshold):
    return len([r for r in results if r > threshold]) / len(results)


def get_l2(fy, gy, x):
    l2 = 0
    for i, dx in enumerate(pairwise(x)):
        x1, x2 = dx
        mf, bf = get_slope_and_intercept(x1, fy[i], x2, fy[i + 1])
        mg, bg = get_slope_and_intercept(x1, gy[i], x2, gy[i + 1])
        l2 += get_l2_component(mf, bf, mg, bg, x2) - get_l2_component(mf, bf, mg, bg, x1)
    return l2


def get_l2_component(mf, bf, mg, bg, x):
    m = mg - mf
    b = bg - bf
    return pow(m, 2) * pow(x, 3) / 3 + m * b * pow(x, 2) + pow(b, 2) * x


def get_slope_and_intercept(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    b = -(y2 - y1) * x2 / (x2 - x1) + y2
    return m, b


def draw_l2_histogram(axes, results, threshold, bin_count):
    axes.hist(x=results, bins=bin_count, color=PLOT_COLORS[0])
    axes.axvline(x=threshold, color=PLOT_COLORS[1], linewidth=PLOT_LINE_WIDTH)
    axes.set_xlabel("L²")
    axes.set_ylabel(OCCURRENCE_COUNT_LABEL)


def draw_p_value_comparison_histogram(axes, results, randomization_counts, bin_count):
    axes.hist(x=results, bins=bin_count, color=PLOT_COLORS, fill=False, histtype="step", label=randomization_counts)
    axes.legend(title="Randomizations")
    axes.set_xlabel(P_VALUE_LABEL)
    axes.set_ylabel(OCCURRENCE_COUNT_LABEL)


def draw_p_value_histogram(axes, results, bin_count, dpi_scale_transform):
    axes.hist(x=results, bins=geomspace(start=0.0005, stop=1, num=bin_count), color=PLOT_COLORS[2])
    text_transform = axes.get_xaxis_transform() + ScaledTranslation(
        xt=-0.0625,
        yt=-0.0625,
        scale_trans=dpi_scale_transform,
    )
    for x, label in [(0.05, "*"), (0.01, "**"), (0.001, "***")]:
        axes.axvline(x=x, color=PLOT_COLORS[1], linewidth=PLOT_LINE_WIDTH)
        axes.text(
            x=x,
            y=1,
            s=label,
            horizontalalignment="right",
            verticalalignment="top",
            transform=text_transform,
        )
    axes.set_xscale("log")
    axes.set_xlabel(P_VALUE_LABEL)
    axes.set_ylabel(OCCURRENCE_COUNT_LABEL)
    axes.invert_xaxis()


def pairwise(iterable):
    a, b = tee(iterable)
    next(b)
    return zip(a, b)
