"""Analyse the evolution of a sample with time."""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tracking_analysis import Analysis


class EmptyDataError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class FolderAnalysis(Analysis):
    """Analyse the folder by time interval."""
    time_interval = 300 #5min in s

    def __init__(self, folder: str) -> None:
        super().__init__(folder)
        if len(self.data) > 0:
            self.time_duration = self.data.time.max()
            self.nb_time_inter = int(1 + self.time_duration //  FolderAnalysis.time_interval)
        else:
            self.time_duration = 0
            self.nb_time_inter = 0

    def group_by_time(self, time_begining: int, time_end: int) -> pd.DataFrame:
        """Group the data by time interval."""
        data = self.data[(self.data.time >= time_begining) & (self.data.time < time_end)]
        lengths = data.chain_length.unique()

        grouped_df = pd.DataFrame()
        grouped = data.groupby("chain_length")

        grouped_df["nb_chains"] = grouped.id.nunique()
        grouped_df["prop_length"] = grouped_df["nb_chains"] / grouped_df["nb_chains"].sum()
        grouped_df[["mean_vel", "std_vel", "min_vel", "max_vel"]] = grouped.velocity.agg(["mean", "std", "min", "max"])
        grouped_df["begining_time"] = time_begining
        grouped_df["end_time"] = time_end
        grouped_df["mean_time"] = (time_begining + time_end) / 2
        grouped_df["chain_length"] = lengths
        return grouped_df

    def plot_evolution_velocity_by_chain_length(self) -> None:
        """Plot the evolution of the mean velocity by chain lengths."""
        plt.figure()
        sns.scatterplot(data=self.grouped_data, x="mean_time", y="mean_vel", hue="chain_length")

        plt.xlabel("Time")
        plt.ylabel("Velocity")
        plt.savefig(os.path.join(self.path, "Figure/velocity_evolution.png"))
        plt.show(block=True)
        plt.close()

    def plot_evolution_nb_chains_by_chain_length(self) -> None:
        """Plot the evolution of the number of chains by chain lengths."""
        plt.figure()
        sns.scatterplot(data=self.grouped_data, x="mean_time", y="prop_length", hue="chain_length")

        plt.xlabel("Time")
        plt.ylabel("Number of chains")

        plt.savefig(os.path.join(self.path, "Figure/nb_chains_evolution.png"))
        plt.show(block=True)
        plt.close()

    def __call__(self) -> None:
        """Process the analysis."""
        if self.nb_time_inter > 0:
            self.calculate_velocity()
            self.calculate_chain_length()
            self.data = self.data.drop(["xBody", "yBody", "bodyMajorAxisLength"], axis=1)
            self.data = Analysis.clean(self.data, self.scale, self.frameRate)
            self.grouped_data = self.group_by_time(0, FolderAnalysis.time_interval)
            for i in range(1, self.nb_time_inter):
                self.grouped_data = pd.concat([self.grouped_data,
                                              self.group_by_time(i * FolderAnalysis.time_interval, (i + 1) * FolderAnalysis.time_interval)], ignore_index=True)
            self.grouped_data.to_csv(os.path.join(self.path, "grouped_data.csv"))
            print(self.grouped_data)
            self.plot_evolution_velocity_by_chain_length()
            self.plot_evolution_nb_chains_by_chain_length()



def main(folder: str) -> int:
    ana = FolderAnalysis(folder)
    ana()


if __name__=="__main__":
    parent_folder = "/Volumes/Chains/ChainFormation"
    folder_list = [os.path.join(parent_folder, f) for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]

    for folder in folder_list[1:]:
        print(folder)
        main(folder)

