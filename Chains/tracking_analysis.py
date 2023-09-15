"""Analysis of the tracking data.""" 

import os
from typing import List

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
        

    def calculate_chain_length(self) -> pd.DataFrame:
        """Calculate the chain length."""
        ids = self.data["id"].unique()
        ids.sort()
        for id in ids:
            data = self.data.loc[self.data["id"] == id]
            mean_length = data["bodyMajorAxisLength"].mean()
            nb_bact = np.rint(mean_length / self.bactLength)
            self.data.loc[self.data["id"] == id, "bactNumber"] = nb_bact

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
        return pd.DataFrame({"id": ids,
                             "bact_number": bact_number,
                             "velocity": mean_vel})

def plot_grouped_data(grouped: pd.DataFrame) -> None:
    """Plot the grouped data."""
    grouped.plot("bact_number", "velocity", "scatter")

    bact_nb = grouped_data["bact_number"].unique()
    print(bact_nb)
    vel: List[float] = []
    std_vel: List[float] = []
    for nb in bact_nb:
        vel.append(grouped.loc[grouped["bact_number"] == nb, "velocity"].mean())
        std_vel.append(grouped.loc[grouped["bact_number"] == nb, "velocity"].std())

    plt.figure()
    plt.errorbar(x=bact_nb, y=vel, yerr=std_vel, linestyle="", marker="s")
    plt.show(block=True)

if __name__ == "__main__":
    folder_path = "/Users/sintes/Desktop/ImageSeq"
    analysis = Analysis(folder_path)
    grouped_data = analysis.process()
    plot_grouped_data(grouped_data)
    