"""Run the simulation of chain swimming."""

import os
from typing import List, Tuple

import numpy as np
from numpy.random import normal
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class ChainGenerator:
    def __init__(self) -> None:
        raise NotImplementedError
    
    def generate(self, n: int) -> float:
        """Generate a chain of length n"""
        raise NotImplementedError

class GaussianChainGenerator(ChainGenerator):
    def __init__(self, mean: float, std: float) -> None:
        """Chain generator where the swimming speed follows a normal distribution"""
        self.mean = mean
        self.std = std

    def generate(self, n: int) -> float:
        """Generate a chain of length n"""
        vel_bacteria: np.ndarray[float] = np.zeros(n)
        for i in range(n):
            vel_bacteria[i] = normal(self.mean, self.std)
        return max(vel_bacteria)

class Simulation:
    def __init__(self, chain_generator: ChainGenerator) -> None:
        self.chain_generator = chain_generator
        self.lengths: List[int] = []
        self.max_vels: List[float] = []

    def generate_chains(self, max_chain_length: int, nb_chain_by_length: int) -> None:
        """Generate chains for the different length"""
        for n in range(1, max_chain_length + 1):
            for _ in range(nb_chain_by_length):
                max_vel = self.chain_generator.generate(n)
                self.lengths.append(n)
                self.max_vels.append(max_vel)
        data = pd.DataFrame({
            "length": self.lengths,
            "max_vel": self.max_vels,
        })
        self.data = data

    def save_data(self, folder, file) -> None:
        """Save the data into a dataframe""" 
        self.data.to_csv(os.path.join(folder, file))

def get_simu_data(data: pd.DataFrame, max_length: int = 8, nb_chains: int = 1000) -> Tuple[Simulation, Simulation]:
    """Run the simulation based on the data mean and std.
    Return the simulation data and the simulation, normalized."""
    data1 = data[data.chain_length==1]
    
    mean = data1.velocity.mean()
    std = data1.velocity.std()

    simu = Simulation(GaussianChainGenerator(mean, std))
    simu_norm = Simulation(GaussianChainGenerator(1, std / mean))
    simu.generate_chains(max_length, nb_chains)
    simu_norm.generate_chains(max_length, nb_chains)   
    return simu, simu_norm

if __name__=="__main__":
    stds = [0.5, 0.68, 1]
    for sd in stds:
        simu = Simulation(GaussianChainGenerator(1, sd))
        simu.generate_chains(10, 10000)

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