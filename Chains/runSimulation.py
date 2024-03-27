"""Run the simulation of chain swimming."""

import os
from typing import List, Tuple

import numpy as np
from numpy.random import normal
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import lognorm

class ChainGenerator:
    def __init__(self) -> None:
        raise NotImplementedError
    
    def sample_vel(self) -> float:
        """Sample a bacteria speed"""
        raise NotImplementedError
    
    def generate(self, n: int) -> float:
        """Generate a chain of length n"""
        vel_bacteria: np.ndarray[float] = np.zeros(n)
        for i in range(n):
            vel_bacteria[i] = self.sample_vel()
        return max(vel_bacteria)

class GaussianChainGenerator(ChainGenerator):
    def __init__(self, mean: float, std: float) -> None:
        """Chain generator where the swimming speed follows a normal distribution"""
        self.mean = mean
        self.std = std

    def sample_vel(self) -> float:
        return normal(self.mean, self.std)
    
class DataChainGenerator(ChainGenerator):
    def __init__(self, data: pd.DataFrame, norm: bool) -> None:
        """Chain generator where the swimming speed follows the experimental distribution."""
        data1 = data[data.chain_length==1]
        if norm:
            vel = data1.Normalized_vel.dropna()
        else:
            vel = data1.velocity.dropna()
        self.vel = np.array(vel)
        self.vel.sort()
        self.sample_size = len(self.vel)


    def sample_vel(self) -> float:
        x = np.random.uniform(0, self.sample_size - 1)
        i = int(np.floor(x))
        p = x - i
        vel = (1 - p) * self.vel[i] + p * self.vel[i + 1]
        return vel

class LogNormalChainGenerator(ChainGenerator):
    """Chain generator where the velocity follows a lognormal distribution"""
    def __init__(self, log_mean: float, log_std: float) -> None:
        self.scale = np.exp(log_mean)
        self.s = log_std

    def sample_vel(self) -> float:
        return lognorm.rvs(s=self.s, scale=self.scale)
    
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

def get_simu_data(data: pd.DataFrame, from_data: bool = False, max_length: int = 8, nb_chains: int = 10000) -> Tuple[Simulation, Simulation]:
    """Run the simulation based on the data mean and std.
    Return the simulation data and the simulation, normalized."""
    data1 = data[data.chain_length==1]
    vel = data1.velocity.dropna()
    mean = vel.mean()
    std = vel.std()
    if from_data:
        simu = Simulation(DataChainGenerator(data, norm=False))
        simu_norm = Simulation(DataChainGenerator(data, norm=True))
    else:
        simu = Simulation(GaussianChainGenerator(mean, std))
        simu_norm = Simulation(GaussianChainGenerator(1, std / mean))
    simu.generate_chains(max_length, nb_chains)
    simu_norm.generate_chains(max_length, nb_chains)   
    return simu, simu_norm

if __name__=="__main__":
    stds = [0.5, 0.68, 1]
    for sd in stds:
        simu = Simulation(LogNormalChainGenerator(1, 1))
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