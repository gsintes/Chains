"""Take the data from all experiments and fuse them."""

import os
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model import calculate_velocity

def get_concentration(concentration_folder: str) -> float:
    """Get the concentration from the folder name."""
    folder_name = concentration_folder.split("/")[-1]
    template = r"Chains\ (\d+.?\d*)\%"
    group = re.match(template, folder_name)
    return float(group[1])

def load_all_data(parent_folder: str) -> pd.DataFrame:
    """Load all the datas from the subfolders."""
    concentration_folders = [os.path.join(parent_folder, f) for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder,f)) and f.startswith("Chains")]
    data = pd.DataFrame()
    for c_folder in concentration_folders:
        concentration = get_concentration(c_folder)
        sub_folders = [os.path.join(c_folder, f) for f in os.listdir(c_folder) if os.path.isdir(os.path.join(c_folder,f))]
        for f in sub_folders:
            sub_data = pd.read_csv(os.path.join(f, "Tracking_Result/vel_data.csv"))
            sub_data["Concentration_LC"] = concentration
            data = pd.concat([data, sub_data], ignore_index=True)
    return data

if __name__ == "__main__":
    parent_folder = "/Users/sintes/Desktop/NASGuillaume/Chains"
    data = load_all_data(parent_folder)

    vel_model = []
    for n in range(1, 11):
        vel_model.append(calculate_velocity(n, 1) / calculate_velocity(1, 1))
    print(vel_model)
    plt.figure()
    sns.scatterplot(data=data, x="bact_number", y="Normalized_vel", hue="Concentration_LC")
    plt.savefig(os.path.join(parent_folder,"Figures/scatter_norm.png"))

    plt.figure()
    sns.scatterplot(data=data, x="bact_number", y="velocity", hue="Concentration_LC")
    plt.savefig(os.path.join(parent_folder,"Figures/scatter_raw.png"))

    plt.figure()
    sns.pointplot(data=data, x="bact_number", y="velocity", hue="Concentration_LC", linestyles="", errorbar="se")
    plt.legend(loc=3)
    plt.savefig(os.path.join(parent_folder,"Figures/errorbar_raw.png"))

    plt.figure()
    sns.pointplot(data=data, x="bact_number", y="Normalized_vel", hue="Concentration_LC", linestyles="", errorbar="se")
    plt.plot(range(1, 11), vel_model, "s", label="Model")
    plt.legend(loc=3)
    plt.savefig(os.path.join(parent_folder,"Figures/errorbar_norm.png"))

    plt.figure()
    sns.pointplot(data=data, x="bact_number", y="velocity", hue="Concentration_LC", linestyles="", errorbar="sd")
    plt.legend(loc=3)
    plt.savefig(os.path.join(parent_folder,"Figures/errorbar_rawsd.png"))

    plt.figure()
    sns.pointplot(data=data, x="bact_number", y="Normalized_vel", hue="Concentration_LC", linestyles="", errorbar="sd")
    plt.plot(range(1, 11), vel_model, "s", label="Model")
    plt.legend(loc=3)
    plt.savefig(os.path.join(parent_folder,"Figures/errorbar_normsd.png"))

    plt.show(block=True)
