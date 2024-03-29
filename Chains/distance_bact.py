"""Calculate the distance between two bacteria over time."""

import os
import sqlite3
from typing import List, Tuple, Any

import numpy as np
import pandas as pd


class DistanceCalculator:
    """Object to calculate distance between pairs of bacteria over time."""
    def __init__(self, path: str) -> None:
        self.bactLength = 10
        self.frameRate = 30
        self.scale = 6.24
        self.path = path
        self.data = self.load_data()
        self.calculate_velocity()
        self.clean()

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
        """Calculate velocities."""
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
    
    def clean(self) -> None:
        """Clean the data by removing to short tracks."""
        ids = self.data["id"].unique()
        for id in ids:
            length: float = self.bactLength * self.data.loc[self.data["id"] == id, "Velocity"].mean() 
            vel: float = self.scale * self.data.loc[self.data["id"] == id, "Velocity"].mean() / self.frameRate #pix/frame
            if vel < 0.2:
                self.data: pd.DataFrame = self.data.drop(self.data[self.data["id"] == id].index)
            else:
                thresh = length / vel
                len_track = len(self.data.loc[self.data["id"] == id])
                if len_track < thresh:
                    self.data = self.data.drop(self.data[self.data["id"] == id].index)
    
    def distance_bacteria(self) -> None:
        """Calculate the distance for all pairs of bacteria"""
        ids = self.data.id.unique()
        ids.sort()
        self.pair_distances = pd.DataFrame(columns=["i", "j", "im", "distance"])
        for k, i in enumerate(ids):
            for j in ids[k + 1:]:
                distances = self.distance_pair(i, j)
                df = pd.DataFrame({
                    "im": [res[0] for res in distances],
                    "distance": [res[1] for res in distances]
                })
                df["i"] = i
                df["j"] = j
                self.pair_distances = pd.concat((self.pair_distances, df))
        self.pair_distances.to_csv(os.path.join(self.path, "Tracking_Result/distances.csv"))

    def distance_pair(self, i: int, j: int) -> List[Tuple[int, float]]:
        """Calculate the distance for all times """
        data = self.data[self.data["id"]==i]
        data = pd.concat((data, self.data[self.data["id"]==j]))
        assert len(data.id.unique()) == 2
        images = list(data["imageNumber"].unique())
        images.sort()
        res = []
        for im in images:
            sub_data = data[data["imageNumber"]==im]
            if len(sub_data) == 2:
                diff = sub_data.diff().dropna()
                distance = np.sqrt(diff.xBody.max() ** 2 + diff.xBody.max() ** 2)
                res.append((im, distance))
            elif len(sub_data) > 2:
                raise ValueError()   
        return res

if __name__=="__main__":
    folder = "/Users/sintes/Desktop/TestDistance"
    calculator = DistanceCalculator(folder)
    calculator.distance_bacteria()
    # print(calculator.pair_distances.i.unique())
