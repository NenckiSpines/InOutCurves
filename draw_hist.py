import numpy as np
import matplotlib.pyplot as plt
import rate_calc

# Function to draw histogram with log scale on x-axis
def draw_histogram(data, bins):
    fig, ax = plt.subplots()
    data_clipped = np.clip(data, a_min=0.0001, a_max=None)
    hist, bin_edges = np.histogram(data_clipped, bins=bins)
    #print (hist,bin_edges)
    max_count = max(hist)
    #print(max_count)
    # Check for non-positive bin edges
    if np.any(bin_edges <= 0):
        raise ValueError("Bin edges must be positive.")

    for i, count in enumerate(hist):
        bin_width = bin_edges[i + 1] - bin_edges[i]
        x = np.log10(bin_edges[i])
        y = 0  # Align bars to the bottom
        width = np.log10(bin_edges[i + 1]) - np.log10(bin_edges[i])
        height = count

        ax.bar(x, height, width=width, bottom=y, align='edge', color='cyan',edgecolor="cyan")

    # Set x-axis ticks and labels
    tick_positions = np.log10([0.1, 0.05, 0.01, 0.001])
    tick_labels = ['0.1', '0.05', '0.01', '0.001']
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram with Log Scale on X-axis')
    
    # Reverse x-axis
    ax.invert_xaxis()
    
    

# Example data
data_WD = np.genfromtxt('p_values_with_differences.csv', delimiter=',')
data_NO = np.genfromtxt('p_values_without_differences.csv', delimiter=',')

bins = np.logspace(np.log10(0.0001), np.log10(1.0), 120)

# Draw histogram
draw_histogram(data_WD, bins)
draw_histogram(data_NO, bins)

rc = rate_calc.RateCalculator()
rc.calculate_rates(data_NO, data_WD)
rc.print_rates()

print(rc.latex_table())
plt.show()