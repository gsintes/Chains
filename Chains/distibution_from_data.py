import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from fused_analysis import load_all_data

from runSimulation import get_simu_data

parent_folder = "/Users/sintes/Desktop/NASGuillaume/Chains"
fig_folder = os.path.join(parent_folder, "Figures")
data = load_all_data(parent_folder)

simu, _ = get_simu_data(data, from_data=True)

for i in range(1, 9):
    data_i = data.loc[data.chain_length==i]
    simu_i = simu.data.loc[simu.data.length==i]
    plt.figure()
    plt.title(f"Chain length = {i}")
    sns.histplot(data=data_i.velocity, stat="density", label="Experiments")
    sns.histplot(data=simu_i.max_vel, stat="density", label="Simulation")
    plt.legend()
    plt.savefig(os.path.join(parent_folder,f"Figures/Velocity_histograms/distib_{i}"))
