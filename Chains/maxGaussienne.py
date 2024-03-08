"""Distribution of the max of the gaussian"""

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

def partion_function(x, mean: float, std: float, n: int) -> float:
    """Return the value of the partition function for the max of n independant variable following N(mean, std)."""
    return (0.5 * (1 + erf((x - mean) / (std * np.sqrt(2))))) ** n

def distribution(x, mean: float, std: float, n: int) -> float:
    """Return the value of the distribution for the max of n independant variable following N(mean, std)."""
    return n * np.exp(-(x - mean) ** 2 / (2 ** std ** 2)) / (std * np.sqrt(2)) * partion_function(x, mean, std, n - 1)

if __name__=="__main__":
    x = np.linspace(0, 15)
    mean = 6
    std = 2
    plt.figure()
    for n in range(1, 9, 2):
        y = partion_function(x, mean, std, n)
        plt.plot(x, y, label=n)

    plt.legend()
    plt.xlabel("Velocity")
    plt.title("Partition function")

    plt.figure()
    for n in range(1, 9, 2):
        y = distribution(x, mean, std, n)
        plt.plot(x, y, label=n)

    plt.legend()
    plt.xlabel("Velocity")
    plt.title("Distribution")
    plt.show(block=True)