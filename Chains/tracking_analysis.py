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
        self.scale = 3.2

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
        self.data["Velocity"] = np.sqrt(self.data["xVel"] ** 2+ self.data["yVel"] ** 2)

    @staticmethod   
    def detect_plateau_value(sequence: pd.Series):
        """Detect a plateau in a serie."""
        try:
            window_size = 10
            list_seq = list(sequence)
            std_moving = sequence.rolling(window_size).std()
            mean = std_moving.mean()
            std = std_moving.std()

            values: List[float] = []
            for i, val_std in enumerate(std_moving):
                if val_std < mean - std:
                    values.append(list_seq[i])
            return np.mean(values)
        except FloatingPointError:
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

    def process(self) -> pd.DataFrame:
        """Performs the analysis."""
        self.calculate_velocity()
        self.calculate_chain_length()
        ids = self.data["id"].unique()
        bact_number: List[int] = []
        mean_vel: List[float] = []
        for id in ids:
            data = self.data.loc[self.data["id"] == id]
            bact_number.append(int(min(data["bactNumber"])))
            mean_vel.append(data["Velocity"].mean())
        self.velocity_data = pd.DataFrame({"id": ids,
                             "bact_number": bact_number,
                             "velocity": mean_vel})
        single_vel = self.velocity_data.loc[self.velocity_data["bact_number"] == 1, "velocity"].mean()
        self.velocity_data["Single_vel"] = single_vel
        self.clean()
        return self.velocity_data

def plot_grouped_data(velocity_data: pd.DataFrame, folder: str) -> None:
    """Plot the grouped data."""
    velocity_data.plot("bact_number", "velocity", "scatter")
    plt.xlabel("Number of bacteria")
    plt.ylabel("Velocity")
    plt.savefig(os.path.join(folder, "Figure/scatter.png"))
    plt.close()

    bact_nb = velocity_data["bact_number"].unique()
    bact_nb.sort()
    vel_l: List[float] = []
    std_vel_l: List[float] = []
    for nb in bact_nb:
        vel_l.append(velocity_data.loc[velocity_data["bact_number"] == nb, "velocity"].mean())
        std_vel_l.append(velocity_data.loc[velocity_data["bact_number"] == nb, "velocity"].std())
    
    vel = np.array(vel_l)
    std_vel = np.array(std_vel_l)

    plt.figure()
    plt.errorbar(x=bact_nb, y=vel / vel[0], yerr=std_vel / vel[0], linestyle="", marker="s")
    plt.xlabel("Number of bacteria")
    plt.ylabel("Velocity")
    plt.savefig(os.path.join(folder, "Figure/error.png"))
    plt.close()


if __name__ == "__main__":
    parent_folder = "/Volumes/Chains/Chains/Chains 12%"
    folder_list: List[str] = [os.path.join(parent_folder, f) for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder,f))]
    folder_list.sort()
    for folder in folder_list:
        print(folder)
        analysis = Analysis(folder)
        velocity_data = analysis.process()
        velocity_data.to_csv(os.path.join(folder,"Tracking_Result/vel_data.csv"), index=None)
        plot_grouped_data(velocity_data, folder)
    