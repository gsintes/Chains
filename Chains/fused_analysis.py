"""Take the data from all experiments and fuse them."""

import os
import re
from typing import List, Tuple
from random import sample

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import model

# mpl.use("pgf")
# plt.rcParams.update({
#     "text.usetex": True,     # use inline math for ticks
#     "pgf.preamble": "\n".join([
#          "\\usepackage{siunitx}",          # load additional packages
#          "\\usepackage{metalogo}",
#          "\\usepackage{unicode-math}",   # unicode math setup
#          ]),
#     "legend.title_fontsize": 20,
#     'xtick.labelsize': 20,
#     'ytick.labelsize': 20
# })

def get_concentration(concentration_folder: str) -> float:
    """Get the concentration from the folder name."""
    folder_name = concentration_folder.split("/")[-1]
    template = r"Chains\ (\d+.?\d*)\%"
    group = re.match(template, folder_name)
    return float(group[1])

def get_date(exp_name: str) -> str:
    """Get the date from the experiment name."""
    return exp_name.split("_")[0]

def normalize_velocity(data: pd.DataFrame) -> pd.DataFrame:
    """Get the normalized velocity using average single vel of the day."""
    for concentration in data["Concentration_LC"].unique():
        sub_data: pd.DataFrame = data.loc[data["Concentration_LC"]==concentration]
        dates = sub_data["Date"].unique()
        for date in dates:
            subdata: pd.DataFrame = data.loc[(data["Date"]==date)&(data["Concentration_LC"]==concentration)&(data["chain_length"]==1)]
            single_vel = subdata["velocity"].mean()
            data.loc[(data["Concentration_LC"]==concentration) & (data["Date"]==date), "Single_vel"] = single_vel
        data["Normalized_vel"] = data["velocity"] / data["Single_vel"]
    return data

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
                sub_data["Date"] = get_date(exp)
                data = pd.concat([data, sub_data], ignore_index=True)
            except FileNotFoundError:
                pass
    data = normalize_velocity(data)
    return data

def velocity_histograms(data: pd.DataFrame, fig_folder: str) -> None:
    """Plot the velocity histograms grouped by chain length."""
    concentrations = data["Concentration_LC"].unique()
    try:
        hist_folder = os.path.join(fig_folder, "Velocity_histograms")
        os.makedirs(hist_folder)
    except FileExistsError:
        pass

    # plt.figure()
    # sns.histplot(data, x="velocity", hue="Concentration_LC", stat="density", common_norm=False)
    # plt.title("Velocities")
    # plt.savefig(os.path.join(hist_folder, "hist_general.png"))
    # plt.close()

    # plt.figure()
    # sns.boxplot(data, y="velocity", hue="Concentration_LC")
    # plt.savefig(os.path.join(hist_folder, "box_plot.png"))
    # plt.close()

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
    c_data = data.copy()
    c_data["n"] = c_data["chain_length"]
    plt.figure()
    g = sns.FacetGrid(c_data, col="n", col_wrap=3, height=2)
    g.map(sns.histplot, "velocity", stat="density", common_norm=False)
    # plt.xlabel(r"$V (\si{\micro\meter\usk\second^{-1}}$")
    plt.savefig(os.path.join(hist_folder, "hist_grouped.png"))
    plt.close()

    for nb in chain_lengths:
        nb_data = c_data.loc[c_data["chain_length"]==nb]
        plt.figure()
        sns.histplot(nb_data, x="velocity", stat="density")
        plt.xlabel(r"$V (\si{\micro\meter\ \second ^{-1}})$", fontsize=20)
        plt.title(f"$n={nb}$", fontsize=20)
        plt.xlim(0, 80)
        plt.tight_layout()
        plt.savefig(os.path.join(hist_folder, f"hist_{nb}.pdf"))
        plt.close()

def size_distribution(data: pd.DataFrame, fig_folder: str) -> None:
    """Plot the chain length distribution."""
    plt.figure()
    sns.histplot(data, x="chain_length", hue="Concentration_LC", multiple="dodge", discrete=True, shrink=0.8)
    plt.xlabel("Chain length")
    plt.savefig(os.path.join(fig_folder, "chain_length_dist_nb.png"))
    plt.close()

    plt.figure()
    sns.histplot(data, x="chain_length", hue="Concentration_LC",
                 multiple="dodge", discrete=True, stat="density", common_norm=False, shrink=0.8)
    plt.xlabel("Chain length")
    plt.savefig(os.path.join(fig_folder, "chain_length_dist_norm.png"))
    plt.close()

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
    plt.close()
    
    plt.figure()
    sns.boxplot(props)
    plt.ylabel("Proportion per direction")
    plt.savefig(os.path.join(fig_folder, "signprop_box.png"))
    plt.close()

def plots_velocity_vs_length(data: pd.DataFrame, fig_folder: str) -> None:
    """Generate the plots velocity vs chain length."""
    plt.figure()
    sns.pointplot(data=data, x="chain_length", y="velocity", linestyles="", errorbar="se", native_scale=True, c="k")
    plt.xlabel(r"$n$", fontsize=20)
    plt.ylabel(r"$V (\si{\micro\meter\ \second ^{-1}})$", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_folder,"errorbar_raw_All.pdf"))
    plt.close()

    plt.figure()
    sns.pointplot(data=data, x="chain_length", y="Normalized_vel", linestyles="", errorbar="se", native_scale=True, c="k")
    plt.xlabel(r"$n$", fontsize=20)
    plt.ylabel(r"$V/V_1$", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_folder,"errorbar_norm_All.pdf"))

    plt.close()

    plt.figure()
    sns.scatterplot(data=data, x="chain_length", y="Normalized_vel", hue="Concentration_LC")
    plt.legend(title="Concentration_LC")
    plt.savefig(os.path.join(fig_folder, "scatter_norm.png"))
    plt.close()

    plt.figure()
    sns.scatterplot(data=data, x="chain_length", y="velocity", hue="Concentration_LC")
    plt.legend(title="Concentration_LC")
    plt.savefig(os.path.join(fig_folder,"scatter_raw.png"))
    plt.close()

    plt.figure()
    sns.pointplot(data=data, x="chain_length", y="velocity", hue="Concentration_LC", linestyles="", errorbar="se", native_scale=True)
    plt.legend(title="Concentration_LC")
    plt.savefig(os.path.join(fig_folder,"errorbar_raw.png"))
    plt.close()

    plt.figure()
    sns.pointplot(data=data, x="chain_length", y="Normalized_vel", hue="Concentration_LC", linestyles="", errorbar="se", native_scale=True)
    plt.legend(title="Concentration_LC")
    plt.savefig(os.path.join(fig_folder, "errorbar_norm.png"))
    plt.close()

    plt.figure()
    sns.pointplot(data=data, x="chain_length", y="velocity", hue="Concentration_LC", linestyles="", errorbar="sd", native_scale=True)
    plt.legend(title="Concentration_LC")
    plt.savefig(os.path.join(fig_folder,"errorbar_rawsd.png"))
    plt.close()

    plt.figure()
    sns.pointplot(data=data, x="chain_length", y="Normalized_vel", hue="Concentration_LC", linestyles="", errorbar="sd", native_scale=True)
    plt.legend(title="Concentration_LC")
    plt.savefig(os.path.join(fig_folder, "errorbar_normsd.png"))
    plt.close()

def plot_vel_by_exp(data: pd.DataFrame, fig_folder: str) -> None:
    """Plot the velocity as a function of chain length grouped by experiment with one plot by concentration of LC."""
    concentrations = data["Concentration_LC"].unique()
    for c in concentrations:
        subdata: pd.DataFrame = data.loc[data["Concentration_LC"]==c]
        plt.figure()
        sns.pointplot(data=subdata, x="chain_length", y="velocity", hue="Exp", linestyles="", errorbar="se", native_scale=True, legend=None)
        plt.savefig(os.path.join(fig_folder, f"groupedExp_{c}%.png"))
        plt.close()
        plt.figure()
        sns.pointplot(data=subdata, x="chain_length", y="Normalized_vel", hue="Exp", linestyles="", errorbar="se", native_scale=True, legend=None)
        plt.savefig(os.path.join(fig_folder, f"Norm_groupedExp_{c}%.png"))
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

def std_plot(data: pd.DataFrame, fig_folder: str) -> None:
    """Plot the standard deviation as a function of length."""
    lengths_exp = data.chain_length.unique()
    stds_exp = []
    for l in lengths_exp:
        sub_data = data[data.chain_length==l]
        stds_exp.append(sub_data.velocity.std())


    plt.figure()
    plt.plot(lengths_exp, stds_exp, "s", label="Experiment")
    plt.xlabel("Chain length")
    plt.ylabel("Standard deviation")
    plt.legend()
    plt.savefig(os.path.join(fig_folder, "Std.png"))
    plt.close()

def sample_size_plot(data: pd.DataFrame, fig_folder: str) -> None:
    """Plot the sample size as a function of the chain length."""
    count = data.chain_length.value_counts()
    plt.figure()
    plt.plot(count, "o")
    plt.yscale("log")

def plot_min(data: pd.DataFrame, fig_folder: str) -> None:
    """Plot the min of the distribution with chain length."""
    lengths_exp = data.chain_length.unique()
    mins_exp: List[float] = []
    for l in lengths_exp:
        sub_data = data[data.chain_length==l]
        mins_exp.append(sub_data.velocity.min())


    plt.figure()
    plt.plot(lengths_exp, mins_exp, "s", label="Experiment")
    plt.xlabel("Chain length")
    plt.ylabel("Minimum velocity")
    plt.legend()
    plt.savefig(os.path.join(fig_folder, "min.png"))
    plt.close()

def plot_max(data: pd.DataFrame, fig_folder: str) -> None:
    """Plot the min of the distribution with chain length."""
    lengths_exp = data.chain_length.unique()
    maxs_exp: List[float] = []
    for l in lengths_exp:
        sub_data = data[data.chain_length==l]
        maxs_exp.append(sub_data.velocity.max())


    plt.figure()
    plt.plot(lengths_exp, maxs_exp, "s", label="Experiment")
    plt.xlabel("Chain length")
    plt.ylabel("Maximum velocity")
    plt.legend()
    plt.savefig(os.path.join(fig_folder, "max.png"))
    plt.close()

def plot_force(data: pd.DataFrame, fig_folder: str) -> None:
    """Plot the force of the chain with chain length."""
    data["force"] = data.velocity / (model.get_A0(data.chain_length) + model.get_A_flagella(model.LENGTH))
    plt.figure()
    sns.pointplot(data=data, x="chain_length", y="force", linestyles="")
    plt.savefig(os.path.join(fig_folder, "force.png"))
    plt.close()

def violin_plot(data: pd.DataFrame, fig_folder: str) -> None:
    """Plot the violin plot of the velocities."""
    plt.figure()

    sns.catplot(data=data, x="chain_length", y="velocity", kind="boxen", native_scale=True)
    # sns.histplot(data=data, x="chain_length", y="velocity")
    sns.pointplot(data=data, x="chain_length", y="velocity", linestyles="", errorbar=None, native_scale=True, c="k")
    plt.ylim((0, 40))
    # plt.xlabel(r"$n$", fontsize=20)
    # plt.ylabel(r"$V (\si{\micro\meter\ \second ^{-1}})$", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_folder, "violin.pdf"))
    plt.close()

if __name__ == "__main__":
    # parent_folder = "/Volumes/Guillaume/Chains"
    # fig_folder = os.path.join(parent_folder, "Figures")
    # data = load_all_data(parent_folder)

    # data.to_csv(os.path.join(parent_folder, "chain_data.csv"))
    data = pd.read_csv("/Users/sintes/Desktop/chain_data.csv")

    data = data[data["chain_length"] <= 8]

    data["Type"] = "Exp"
    simu = pd.read_csv("/Users/sintes/Desktop/FromDataEvenSpacing/data.csv")
    simu = simu[simu["chain_length"] <= 8]
    simu["Type"] = "Simu"
    simu = simu[simu.step <=500]
    simu["velocity"] = simu["vel"]
    simu = simu.drop_duplicates(("Simu_nb", "id"))
    data=pd.concat([data, simu], ignore_index=True)
    data=data[["chain_length", "velocity", "Type"]]

    plt.figure()
    sns.violinplot(data=data, x="chain_length", y="velocity", hue="Type", split=True, native_scale=True, inner=None, fill=False, cut=0)
    # sns.stripplot(data=data, x="chain_length", y="velocity", hue="Type", native_scale=True)
    sns.pointplot(data=data, x="chain_length", y="velocity", hue="Type", linestyles="", errorbar=None, native_scale=True, dodge=0.2)
    plt.show(block=True)
    # plot_force(data, fig_folder)
    # plot_proportion_sign(data, fig_folder)
    # plots_velocity_vs_length(data, fig_folder)
    # violin_plot(data, fig_folder)
    # velocity_histograms(data, fig_folder)
    # size_distribution(data, fig_folder)
    # plot_vel_by_exp(data, fig_folder)
    # std_plot(data, fig_folder)
    # sample_size_plot(data, fig_folder)
    # plot_min(data, fig_folder)
    # plot_max(data, fig_folder)
