import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def plot_dist(arr):
    sorted_arr = sorted(arr)
    fit = norm.pdf(sorted_arr, np.mean(sorted_arr), np.std(sorted_arr))
    plt.figure("Age distribution")
    plt.plot(sorted_arr, fit, "-o")
    plt.hist(sorted_arr, bins=20, density=True)
    plt.show()
