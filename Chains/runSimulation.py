"""Run the simulation of chain swimming."""

from typing import List

import numpy as np
from numpy.random import normal
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Chain:
    def __init__(self, n: int, mean: float, std: float) -> None:
        """Generate a chain from the swimming speed normal distribution"""
        self.length = n
        self.vel_bacteria: np.ndarray[float] = np.zeros(self.length)
        for i in range(self.length):
            self.vel_bacteria[i] = normal(mean, std)
        self.max_vel = max(self.vel_bacteria)
    
class run_simulation:
    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std
        self.length: List[int] = []
        self.max_vels: List[float] = []

    def generate_chains(self, max_chain_length: int, nb_chain_by_length: int) -> None:
        """Generate chains for the different length"""
        for n in range(1, max_chain_length + 1):
            for _ in range(nb_chain_by_length):
                chain = Chain(n, self.mean, self.std)
                self.length.append(n)
                self.max_vels.append(chain.max_vel)
    
    def save_data(self) -> None:
        """Save the data into a dataframe"""
        data = pd.DataFrame({
            "length": self.length,
            "max_vel": self.max_vels,
        })
        self.data = data
        # data.to_csv("/Users/sintes/Desktop/NASGuillaume/Chains/Simu/Chain_vel.csv")

if __name__=="__main__":
    stds = [0.5, 0.68, 1]
    for sd in stds:
        simu = run_simulation(1, sd)
        simu.generate_chains(10, 10000)
        simu.save_data()

        sns.pointplot(data=simu.data, x="length", y="max_vel", linestyles="", errorbar="sd", native_scale=True, label=sd)
    n = np.arange(1,11)
    plt.plot(n, 1 + np.sqrt(np.log(n)), label="$log(n)^{0.5}$")
    plt.legend()
 

    data = simu.data
    lengths = data.length.unique()
    stds = []
    for l in lengths:
        sub_data = data[data.length==l]
        stds.append(sub_data.max_vel.std())
    plt.figure()
    plt.loglog(lengths, stds, "o")
    plt.xlabel("Chain length")
    plt.ylabel("Standard deviation")

    plt.show(block=True)