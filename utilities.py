from itertools import tee

from matplotlib import pyplot
from matplotlib.pyplot import hist, axvline


def get_p_value(results, threshold):
    return len([r for r in results if r >= threshold]) / len(results)


def get_l2(fy, gy, x):
    l2 = 0
    for i, dx in enumerate(pairwise(x)):
        x1, x2 = dx
        mf, bf = get_slope_and_intercept(x1, fy[i], x2, fy[i + 1])
        mg, bg = get_slope_and_intercept(x1, gy[i], x2, gy[i + 1])
        l2 += get_l2_component(mf, bf, mg, bg, x2) - get_l2_component(mf, bf, mg, bg, x1)
    return l2


def get_l2_component(mf, bf, mg, bg, x):
    return pow(x * (mg - mf) + bg - bf, 3) / (3 * (mg - mf))


def get_slope_and_intercept(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    b = -(y2 - y1) * x2 / (x2 - x1) + y2
    return m, b


def draw_l2_histogram(results, threshold, bin_count, title):
    pyplot.title(title)
    hist(results, bin_count)
    axvline(x=threshold, c="r")


def pairwise(iterable):
    a, b = tee(iterable)
    next(b)
    return zip(a, b)

