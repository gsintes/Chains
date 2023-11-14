"""Analysis of the tracking data.""" 

import os
from typing import List

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')

class Analysis():
    """Class to perform the analysis of chain tracking data."""
    def __init__(self, path) -> None:
        self.bactLength = 10
        self.frameRate = 30
        self.scale = 6.24
        self.path = path
        self.data = self.load_data()

    def load_data(self) -> pd.DataFrame:
        """Load the data from the database."""
        dbfile = os.path.join(self.path, "Tracking_Result/tracking.db")
        con = sqlite3.connect(dbfile)
        df = pd.read_sql_query('SELECT * FROM tracking', con)
        con.close()
        df = df[['xBody', 'yBody', 'tBody', 'bodyMajorAxisLength', 'bodyMinorAxisLength', 'imageNumber', 'id']]
        df["time"] = df["imageNumber"] / self.frameRate 
        return df

    def calculate_velocity(self) -> None:
        """Reindex the objects when big jumps."""
        ids = self.data["id"].unique()
        ids.sort()
        for id in ids:
            data = self.data.loc[self.data["id"] == id]
            coord = ["xBody", "yBody"]
            for ax in coord:
                pos_diff = data[ax].diff() / self.scale
                time_diff = data["time"].diff()
                
                velocity = pos_diff / time_diff
                self.data.loc[velocity.index, ax[0] + "Vel"] = velocity
        self.data["Velocity"] = np.sqrt(self.data["xVel"] ** 2 + self.data["yVel"] ** 2)

    @staticmethod   
    def detect_plateau_value(sequence: pd.Series):
        """Detect a plateau in a serie."""

        window_size = 10
        list_seq = list(sequence)
        std_moving = sequence.rolling(window_size).std()
        mean = std_moving.mean()
        std = std_moving.std()

        values: List[float] = []
        for i, val_std in enumerate(std_moving):
            if val_std < mean - std:
                values.append(list_seq[i])
        if len(values) != 0:    
            return np.mean(values)
        else:
            return sequence.mean()

    def calculate_chain_length(self) -> pd.DataFrame: 
        """Calculate the chain length."""
        ids = self.data["id"].unique()
        ids.sort()
        for id in ids:
            data = self.data.loc[self.data["id"] == id]
            mean_length = analysis.detect_plateau_value(data["bodyMajorAxisLength"]) 
            nb_bact = np.rint(mean_length / self.bactLength)
            self.data.loc[self.data["id"] == id, "bactNumber"] = nb_bact

    def clean(self) -> None:
        """Clean the data by removing to short tracks."""
        ids = self.data["id"].unique()
        for id in ids:
            length: float = self.bactLength * self.velocity_data.loc[self.velocity_data["id"] == id, "velocity"].mean() 
            vel: float = self.scale * self.velocity_data.loc[self.velocity_data["id"] == id, "velocity"].mean() / self.frameRate #pix/frame
            if vel < 0.2:
                self.velocity_data: pd.DataFrame = self.velocity_data.drop(self.velocity_data[self.velocity_data["id"] == id].index)
            else:
                thresh = length / vel
                len_track = len(self.data.loc[self.data["id"] == id])
                if len_track < thresh:
                    self.velocity_data = self.velocity_data.drop(self.velocity_data[self.velocity_data["id"] == id].index)

    def trace_image(self) -> None:
        """Draw the image with superimposed trajectories."""
        plt.figure()
        plt.xlim((0, 1024))
        plt.ylim((0, 1024))
        ids = self.data["id"].unique()
        for id in ids:
            sub_data: pd.DataFrame = self.data.loc[self.data["id"]==id]
            plt.plot(sub_data["xBody"], 1024 - sub_data["yBody"], ".", markersize=1)
        plt.savefig(os.path.join(self.path, "Figure/trace.png"))

    def process(self) -> pd.DataFrame:
        """Performs the analysis."""
        self.trace_image()
        self.calculate_velocity()
        self.calculate_chain_length()
        ids = self.data["id"].unique()
        chain_length: List[int] = []
        mean_vel: List[float] = []
        for id in ids:
            data = self.data.loc[self.data["id"] == id]
            chain_length.append(int(min(data["bactNumber"])))
            mean_vel.append(data["Velocity"].mean())
        self.velocity_data = pd.DataFrame({"id": ids,
                             "chain_length": chain_length,
                             "velocity": mean_vel})
        self.clean()
        single_vel = self.velocity_data.loc[self.velocity_data["chain_length"] == 1, "velocity"].mean()
        self.velocity_data["Single_vel"] = single_vel
        self.velocity_data["Normalized_vel"] = self.velocity_data["velocity"] / single_vel
        return self.velocity_data

def plot_grouped_data(velocity_data: pd.DataFrame, folder: str) -> None:
    """Plot the grouped data."""
    velocity_data.plot("chain_length", "velocity", "scatter")
    plt.xlabel("Number of bacteria")
    plt.ylabel("V")
    plt.savefig(os.path.join(folder, "Figure/scatter_raw.png"))
    plt.close()

    velocity_data.plot("chain_length", "velocity", "scatter")
    plt.xlabel("Number of bacteria")
    plt.ylabel("$V/V_0$")
    plt.savefig(os.path.join(folder, "Figure/scatter_norm.png"))
    plt.close()

    bact_nb = velocity_data["chain_length"].unique()
    bact_nb.sort()
    vel_l: List[float] = []
    se_vel_l: List[float] = []
    for nb in bact_nb:
        vel_l.append(velocity_data.loc[velocity_data["chain_length"] == nb, "velocity"].mean())
        se_vel_l.append(velocity_data.loc[velocity_data["chain_length"] == nb, "velocity"].sem()) #standard error
    
    vel = np.array(vel_l)
    se_vel = np.array(se_vel_l)

    plt.figure()
    plt.errorbar(x=bact_nb, y=vel / vel[0], yerr=se_vel / vel[0], linestyle="", marker="s")
    plt.xlabel("Number of bacteria")
    plt.ylabel("Velocity")
    plt.savefig(os.path.join(folder, "Figure/error_norm.png"))
    plt.close()

    plt.figure()
    plt.errorbar(x=bact_nb, y=vel, yerr=se_vel, linestyle="", marker="s")
    plt.xlabel("Number of bacteria")
    plt.ylabel("Velocity")
    plt.savefig(os.path.join(folder, "Figure/error_raw.png"))
    plt.close()



if __name__ == "__main__":
    parent_folder = "/Users/sintes/Desktop/TestVideosChains"
    folder_list: List[str] = [os.path.join(parent_folder, f) for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder,f))]
    folder_list.sort()
    for folder in folder_list:
        print(folder)
        analysis = Analysis(folder)
        if len(analysis.data) > 0:
            velocity_data = analysis.process()
            velocity_data.to_csv(os.path.join(folder,"Tracking_Result/vel_data.csv"), index=None)
            plot_grouped_data(velocity_data, folder)
