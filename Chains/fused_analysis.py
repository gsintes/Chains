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
            exp = f.split("/")[-1]
            sub_data = pd.read_csv(os.path.join(f, "Tracking_Result/vel_data.csv"))
            sub_data["Concentration_LC"] = concentration
            sub_data["Exp"] = exp
            data = pd.concat([data, sub_data], ignore_index=True)
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

    for c in concentrations:
        try:
            c_folder = os.path.join(hist_folder, str(c))
            os.makedirs(c_folder)
        except FileExistsError:
            pass
        c_data: pd.DataFrame = data.loc[data["Concentration_LC"] == c]
        bact_numbers = c_data["bact_number"].unique()
        bact_numbers.sort()
        plt.figure()
        sns.histplot(c_data, x="velocity", stat="density")
        plt.title("Velocities")
        plt.savefig(os.path.join(c_folder, "hist_general.png"))
        plt.close()

        for nb in bact_numbers:
            nb_data = c_data.loc[c_data["bact_number"]==nb]
            data_length = len(nb_data)
            plt.figure()
            sns.histplot(nb_data, x="velocity", stat="density")
            plt.title(f"Chain length : {nb}")
            plt.savefig(os.path.join(c_folder, f"hist_{nb}.png"))
            plt.close()

def size_distribution(data: pd.DataFrame, fig_folder: str) -> None:
    """Plot the chain length distribution."""
    concentrations = data["Concentration_LC"].unique()
    
    plt.figure()
    sns.histplot(data, x="bact_number", hue="Concentration_LC", multiple="dodge", discrete=True, shrink=0.8)
    plt.xlabel("Chain length")
    plt.savefig(os.path.join(fig_folder, "chain_length_dist_nb.png")) 
    plt.figure()
    sns.histplot(data, x="bact_number", hue="Concentration_LC",
                 multiple="dodge", discrete=True, stat="density", common_norm=False, shrink=0.8)
    plt.xlabel("Chain length")
    plt.savefig(os.path.join(fig_folder, "chain_length_dist_norm.png")) 

def plots_velocity_number(data: pd.DataFrame, fig_folder: str) -> None:
    """Generate the plots velocity vs chain length."""
    vel_model = []
    for n in range(1, 11):
        vel_model.append(calculate_velocity(n, 1) / calculate_velocity(1, 1))

    plt.figure()
    sns.scatterplot(data=data, x="bact_number", y="Normalized_vel", hue="Concentration_LC")
    plt.savefig(os.path.join(fig_folder, "scatter_norm.png"))
    plt.close()

    plt.figure()
    sns.scatterplot(data=data, x="bact_number", y="velocity", hue="Concentration_LC")
    plt.savefig(os.path.join(fig_folder,"scatter_raw.png"))
    plt.close()

    plt.figure()
    sns.pointplot(data=data, x="bact_number", y="velocity", hue="Concentration_LC", linestyles="", errorbar="se")
    plt.legend(loc=3)
    plt.savefig(os.path.join(fig_folder,"errorbar_raw.png"))
    plt.close()

    plt.figure()
    sns.pointplot(data=data, x="bact_number", y="Normalized_vel", hue="Concentration_LC", linestyles="", errorbar="se")
    plt.plot(range(1, 11), vel_model, "s", label="Model")
    plt.legend(loc=3)
    plt.savefig(os.path.join(fig_folder, "errorbar_norm.png"))
    plt.close()

    plt.figure()
    sns.pointplot(data=data, x="bact_number", y="velocity", hue="Concentration_LC", linestyles="", errorbar="sd")
    plt.legend(loc=3)
    plt.savefig(os.path.join(fig_folder,"errorbar_rawsd.png"))
    plt.close()

    plt.figure()
    sns.pointplot(data=data, x="bact_number", y="Normalized_vel", hue="Concentration_LC", linestyles="", errorbar="sd")
    plt.plot(range(1, 11), vel_model, "s", label="Model")
    plt.legend(loc=3)
    plt.savefig(os.path.join(fig_folder, "errorbar_normsd.png"))
    plt.close()

if __name__ == "__main__":
    parent_folder = "/Users/sintes/Desktop/NASGuillaume/Chains"
    fig_folder = os.path.join(parent_folder, "Figures")
    data = load_all_data(parent_folder)

    data.to_csv(os.path.join(parent_folder, "chain_data.csv"))

    plots_velocity_number(data, fig_folder)
    velocity_histograms(data, fig_folder)
    size_distribution(data, fig_folder)
