"""Calculate the distance between two bacteria over time."""

import os
from datetime import datetime
import sqlite3
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import time

def timeit(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result

    return timed

def load_data(path: str, frame_rate: int) -> pd.DataFrame:
        """Load the data from the database."""
        dbfile = os.path.join(path, "Tracking_Result/tracking.db")
        con = sqlite3.connect(dbfile)
        df = pd.read_sql_query('SELECT * FROM tracking', con)
        con.close()
        df = df[['xBody', 'yBody', 'tBody', 'bodyMajorAxisLength', 'bodyMinorAxisLength', 'imageNumber', 'id']]
        df["time"] = df["imageNumber"] / frame_rate 
        return df

class DistanceCalculator:
    """Object to calculate distance between pairs of bacteria over time."""
    def __init__(self, path: str) -> None:
        self.bactLength = 10
        self.frame_rate = 30
        self.scale = 6.24
        self.path = path
        self.data = load_data(path, self.frame_rate)
        self.calculate_velocity()
        self.clean()
    
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
            vel: float = self.scale * self.data.loc[self.data["id"] == id, "Velocity"].mean() / self.frame_rate #pix/frame
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
        res = []
        for k, i in enumerate(ids):
            for j in ids[k + 1:]:
                dist = self.distance_pair(i, j)
                for d in dist:
                    res.append([i, j, d[0], d[1]])
        if len(res) > 0:
            res = np.array(res)
            self.pair_distances = pd.DataFrame({
                "i": res[:, 0],
                "j": res[:, 1],
                "im": res[:, 2],
                "distance": res[:, 3]
            })
        else:
            self.pair_distances = pd.DataFrame()
        self.pair_distances.dropna(inplace=True)
        self.pair_distances.to_csv(os.path.join(self.path, "Tracking_Result/distances.csv"), index=False)

    def distance_pair(self, i: int, j: int) -> List[Tuple[int, float]]:
        """Calculate the distance for all times """
        data = self.data[self.data["id"]==i]
        data = pd.concat((data, self.data[self.data["id"]==j]))
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

class DistanceAnalyser:
    """Analyse the distance between to objects"""
    def __init__(self, path: str, distance: pd.DataFrame, tracking: pd.DataFrame, i: int, j: int) -> None:
        self.path = path
        self.tracking = tracking
        self.distance = distance[distance["i"]==i]
        self.distance = self.distance[self.distance["j"]==j]
        self.i = i
        self.j = j

    def process(self) -> bool:
        "Run the distance analysis."
        pot = self.potential_fusion()
        if pot:
            self.plot_distance()
        return pot
    
    def last_disparition(self) -> int:
        """Check if both particles disappear simultenaously"""
        self.track_i = self.tracking[self.tracking["id"==self.i]]
        self.track_j = self.tracking[self.tracking["id"==self.i]]
        last_im_i = self.track_i.imageNumber.max()
        last_im_j = self.track_j.imageNumber.max()
        if (last_im_i - last_im_j) > 0:
            return self.i
        if last_im_j < last_im_j:
            return self.j
        else:
            return 0

    def potential_fusion(self) -> bool:
        """Check if there is a potential fusion of the two bacteria."""
        if len(self.distance) > 60:
            end_distance = self.distance.distance[-30:].min()
            if end_distance < 20:
                if self.distance.distance[-60: -30].mean() > end_distance:
                    print(self.last_disparition())

                    return True
        return False
    
    def plot_distance(self):
        """Plot the distance as a function of time"""
        plt.figure()
        plt.plot(self.distance["im"], self.distance["distance"], ".")
        plt.ylim(bottom=0)
        plt.xlabel("Image")
        plt.ylabel("Distance (pixel)")
        plt.title(f"Pair: ({self.i}, {self.j})")
        plt.savefig(os.path.join(self.path, f"Figure/Distance/{self.i}-{self.j}.png"))
        plt.close()

if __name__=="__main__":
    parent_folder = "/home/guillaume/NAS/Chains/Chains 12%"
    folder_list: List[str] = [os.path.join(parent_folder, f) for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder,f))]
    folder_list.sort()

    folders = []
    i_list = []
    j_list = []
    last_ims = []
    log_file = os.path.join(parent_folder, "log.txt")
    with open(log_file, "w") as file:
        file.write("Distance code \n")

    for folder in folder_list:
        try:
            os.makedirs(os.path.join(folder, "Figure/Distance"))
        except FileExistsError:
            pass
        try:
            calculator = DistanceCalculator(folder)
            calculator.distance_bacteria()
        
            pairs = calculator.pair_distances.groupby(['i','j']).count().reset_index()[["i", "j"]]
            for pair in pairs.iterrows():
                ana = DistanceAnalyser(folder, calculator.pair_distances, calculator.data, pair[1].i, pair[1].j)
                if ana.process():
                    folders.append(folder)
                    i_list.append(ana.i)
                    j_list.append(ana.j)
                    last_ims.append(ana.distance.im.max())
        except KeyError:
            pass
        with open(log_file, 'a') as file:
            file.write(f"{folder} done at {datetime.now()}\n")
    pot_fusion = pd.DataFrame({
        "folder": folders,
        "i": i_list,
        "j": j_list,
        "last_im": last_ims,
        "checked": [0 for i in folders]
    })
    pot_fusion.to_csv(os.path.join(parent_folder, "Potential_fusion.csv"))
