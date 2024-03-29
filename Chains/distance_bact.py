"""Calculate the distance between two bacteria over time."""

import os
import sqlite3
from typing import List, Tuple, Any

import numpy as np
import pandas as pd

from tracking_analysis import Analysis

class DistanceCalculator:
    """Object to calculate distance between pairs of bacteria over time."""
    def __init__(self, path: str) -> None:
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
