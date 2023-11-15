"""Take the data from all experiments and fuse them."""

import os
import re
from typing import List, Tuple
from random import sample

import pandas as pd
import numpy as np
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
            exp = f.split("/")[-1]
            try:
                sub_data = pd.read_csv(os.path.join(f, "Tracking_Result/vel_data.csv"))
                sub_data["Concentration_LC"] = concentration
                sub_data["Exp"] = exp
                data = pd.concat([data, sub_data], ignore_index=True)
            except FileNotFoundError:
                pass
    return data

def velocity_histograms(data: pd.DataFrame, fig_folder: str) -> None:
    """Plot the velocity histograms grouped by chain length."""
    concentrations = data["Concentration_LC"].unique()
    try:
        hist_folder = os.path.join(fig_folder, "Velocity_histograms")
        os.makedirs(hist_folder)
    except FileExistsError:
        pass

    plt.figure()
    sns.histplot(data, x="velocity", hue="Concentration_LC", stat="density", common_norm=False)
    plt.title("Velocities")
    plt.savefig(os.path.join(hist_folder, "hist_general.png"))

    plt.figure()
    sns.boxplot(data, y="velocity", hue="Concentration_LC")
    plt.savefig(os.path.join(hist_folder, "box_plot.png"))

    for c in concentrations:
        try:
            c_folder = os.path.join(hist_folder, str(c))
            os.makedirs(c_folder)
        except FileExistsError:
            pass
        c_data: pd.DataFrame = data.loc[data["Concentration_LC"] == c]
        chain_lengths = c_data["chain_length"].unique()
        chain_lengths.sort()
        plt.figure()
        sns.histplot(c_data, x="velocity", stat="density")
        plt.title("Velocities")
        plt.savefig(os.path.join(c_folder, "hist_general.png"))
        plt.close()

        plt.figure()
        g = sns.FacetGrid(c_data, col="chain_length", col_wrap=3, height=2)
        g.map(sns.histplot, "velocity", stat="density", common_norm=False)
        plt.savefig(os.path.join(c_folder, "hist_grouped.png"))
        plt.close()

        for nb in chain_lengths:
            nb_data = c_data.loc[c_data["chain_length"]==nb]
            plt.figure()
            sns.histplot(nb_data, x="velocity", stat="density")
            plt.title(f"Chain length : {nb}")
            plt.savefig(os.path.join(c_folder, f"hist_{nb}.png"))
            plt.close()

def size_distribution(data: pd.DataFrame, fig_folder: str) -> None:
    """Plot the chain length distribution."""    
    plt.figure()
    sns.histplot(data, x="chain_length", hue="Concentration_LC", multiple="dodge", discrete=True, shrink=0.8)
    plt.xlabel("Chain length")
    plt.savefig(os.path.join(fig_folder, "chain_length_dist_nb.png")) 

    plt.figure()
    sns.histplot(data, x="chain_length", hue="Concentration_LC",
                 multiple="dodge", discrete=True, stat="density", common_norm=False, shrink=0.8)
    plt.xlabel("Chain length")
    plt.savefig(os.path.join(fig_folder, "chain_length_dist_norm.png")) 

def plot_proportion_sign(data: pd.DataFrame, fig_folder: str) -> None:
    """Plot the proportion of bacteria swimming in one direction."""
    exps = data["Exp"].unique()
    props: List[float] = []
    for exp in exps:
        sub_data = data.loc[data["Exp"]==exp]
        prop = len(sub_data[sub_data["sign"]== 1]) / len(sub_data)
        props.append(max(prop, 1 - prop))
    plt.figure()
    plt.plot(np.ones(len(props)), props, "k.")
    plt.ylabel("Proportion per direction")
    plt.savefig(os.path.join(fig_folder, "signprop_points.png"))

    plt.figure()
    sns.boxplot(props)
    plt.ylabel("Proportion per direction")
    plt.savefig(os.path.join(fig_folder, "signprop_box.png"))

def plots_velocity_vs_length(data: pd.DataFrame, fig_folder: str) -> None:
    """Generate the plots velocity vs chain length."""
    vel_model = []
    for n in range(1, 11):
        vel_model.append(calculate_velocity(n, 1) / calculate_velocity(1, 1))

    plt.figure()
    sns.scatterplot(data=data, x="chain_length", y="Normalized_vel", hue="Concentration_LC")
    plt.savefig(os.path.join(fig_folder, "scatter_norm.png"))
    plt.close()

    plt.figure()
    sns.scatterplot(data=data, x="chain_length", y="velocity", hue="Concentration_LC")
    plt.savefig(os.path.join(fig_folder,"scatter_raw.png"))
    plt.close()

    plt.figure()
    sns.pointplot(data=data, x="chain_length", y="velocity", hue="Concentration_LC", linestyles="", errorbar="se", native_scale=True)
    plt.legend()
    plt.savefig(os.path.join(fig_folder,"errorbar_raw.png"))
    plt.close()

    plt.figure()
    sns.pointplot(data=data, x="chain_length", y="Normalized_vel", hue="Concentration_LC", linestyles="", errorbar="se", native_scale=True)
    plt.plot(range(1, 11), vel_model, "s", label="Model")
    plt.legend()
    plt.savefig(os.path.join(fig_folder, "errorbar_norm.png"))
    plt.close()

    plt.figure()
    sns.pointplot(data=data, x="chain_length", y="velocity", hue="Concentration_LC", linestyles="", errorbar="sd", native_scale=True)
    plt.legend()
    plt.savefig(os.path.join(fig_folder,"errorbar_rawsd.png"))
    plt.close()

    plt.figure()
    sns.pointplot(data=data, x="chain_length", y="Normalized_vel", hue="Concentration_LC", linestyles="", errorbar="sd", native_scale=True)
    plt.plot(range(1, 11), vel_model, "s", label="Model")
    plt.legend()
    plt.savefig(os.path.join(fig_folder, "errorbar_normsd.png"))
    plt.close()

def sampling(data: pd.DataFrame, nb: int = 10) -> List[Tuple[str, int, int]]:
    """Give a sampling of the data to check detection."""
    lengths = data["chain_length"].unique()
    lengths.sort()
    res: List[Tuple[str, int, int]] = []
    for l in lengths:
        sub_data: pd.DataFrame = data.loc[data["chain_length"]==l]
        if len(sub_data) > nb:
            indexes = list(sub_data.index)
            subset = sample(indexes, nb)
            sub_data = sub_data.loc[subset]
        
        for chain in sub_data.iterrows():
            res.append((chain[1]["Exp"], chain[1]["id"], l))
    return res

if __name__ == "__main__":
    parent_folder = "/Users/sintes/Desktop/NASGuillaume/Chains"
    fig_folder = os.path.join(parent_folder, "Figures")
    data = load_all_data(parent_folder)

    data.to_csv(os.path.join(parent_folder, "chain_data.csv"))

    plot_proportion_sign(data, fig_folder)
    plots_velocity_vs_length(data, fig_folder)
    velocity_histograms(data, fig_folder)
    size_distribution(data, fig_folder)
   