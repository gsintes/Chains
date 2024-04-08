"""Calculate the distance between two bacteria over time."""

import os
from datetime import datetime
import sqlite3
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
BACTLENGTH = 10
FRAME_RATE = 30
SCALE = 6.24


def load_data(path: str, frame_rate: int) -> pd.DataFrame:
        """Load the data from the database."""
        dbfile = os.path.join(path, "Tracking_Result/tracking.db")
        con = sqlite3.connect(dbfile)
        df = pd.read_sql_query('SELECT * FROM tracking', con)
        con.close()
        df = df[['xBody', 'yBody', 'tBody', 'bodyMajorAxisLength', 'bodyMinorAxisLength', 'imageNumber', 'id']]
        df["time"] = df["imageNumber"] / frame_rate 
        return df
   
def detect_plateau_value(sequence: pd.Series) -> float:
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

def get_apparition(data: pd.DataFrame) -> List[Tuple[int, int]]:
    """Detect the apparition for each ids."""
    ids = data.id.unique()
    res = []
    for id in ids:
        sub_data = data[data.id==id]
        res.append((int(id), int(sub_data.imageNumber.min())))
    return res

def clean(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the data by removing to short tracks."""
    ids = data["id"].unique()
    for id in ids:
        length: float = BACTLENGTH * data.loc[data["id"] == id, "Velocity"].mean() 
        vel: float = SCALE * data.loc[data["id"] == id, "Velocity"].mean() / FRAME_RATE#pix/frame
        if vel < 0.2:
            data: pd.DataFrame = data.drop(data[data["id"] == id].index)
        else:
            thresh = length / vel
            len_track = len(data.loc[data["id"] == id])
            if len_track < thresh:
                data = data.drop(data[data["id"] == id].index)
    return data.copy()

class DistanceCalculator:
    """Object to calculate distance between pairs of bacteria over time."""
    def __init__(self, path: str) -> None:
        
        self.path = path
        self.data = load_data(path, self.frame_rate)
        self.calculate_velocity()
        self.data = clean(self.data)
    
    def calculate_velocity(self) -> None:
        """Calculate velocities."""
        ids = self.data["id"].unique()
        ids.sort()
        for id in ids:
            data = self.data.loc[self.data["id"] == id]
            coord = ["xBody", "yBody"]
            for ax in coord:
                pos_diff = data[ax].diff() / SCALE
                time_diff = data["time"].diff()
                
                velocity = pos_diff / time_diff
                self.data.loc[velocity.index, ax[0] + "Vel"] = velocity
        self.data["Velocity"] = np.sqrt(self.data["xVel"] ** 2 + self.data["yVel"] ** 2)
    
    
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
    def __init__(self, path: str, distance: pd.DataFrame, tracking: pd.DataFrame, apparitions, i: int, j: int) -> None:
        self.path = path
        self.tracking = tracking
        self.distance = distance[distance["i"]==i]
        self.distance = self.distance[self.distance["j"]==j]
        self.i = i
        self.j = j
        self.apparitions = apparitions

    def process(self) -> bool:
        "Run the distance analysis."
        pot = self.potential_fusion()
        if pot:
            self.plot_distance()
        return pot
    
    def last_disparition(self) -> str:
        """Check if both particles disappear simultenaously"""
        self.track_i = self.tracking[self.tracking["id"]==self.i]
        self.track_j = self.tracking[self.tracking["id"]==self.j]
        last_im_i = self.track_i.imageNumber.max()
        last_im_j = self.track_j.imageNumber.max()
        if (last_im_i - last_im_j) > 0:
            return "i"
        if last_im_j < last_im_j:
            return "j"
        else:
            return ""
        
    def size_increase(self, remaining) -> bool:
        """Check if the remaining particles increases in size."""
        if remaining == "i":
            remaining_track = self.track_i
            disparu = self.track_j
        else:
            remaining_track = self.track_j
            disparu = self.track_i
        size_disparu = detect_plateau_value(disparu.bodyMajorAxisLength)
        previous_size = detect_plateau_value(remaining_track[remaining_track["imageNumber"] < self.last_im].bodyMajorAxisLength)
        new_size = detect_plateau_value(remaining_track[remaining_track["imageNumber"] > self.last_im].bodyMajorAxisLength)
        delta_size = np.abs(new_size - previous_size)
        return  delta_size / size_disparu < 0.1
    
    def apparition_to_check(self) -> List[int]:
        """Check if a chain appears after a disparition."""
        res: List[int] = []
        for app in self.apparitions:
            if self.last_im - 10 < app[1] < self.last_im + 10:
                res.append(app[0])
        return res

    def check_apparition(self, id) -> bool:
        """Check if the new chain could be from the fusion."""
        sub_data: pd.DataFrame = self.tracking[self.tracking.id ==id]
        sub_data.sort_values(by="imageNumber", inplace=True, ignore_index=True)
        size_i = detect_plateau_value(self.track_i.bodyMajorAxisLength)
        size_j = detect_plateau_value(self.track_j.bodyMajorAxisLength)
        size_sum = size_i + size_j
        size_new = detect_plateau_value(sub_data.bodyMajorAxisLength)
        if np.abs(size_new - size_sum) / size_sum < 0.1:
            final_x = self.track_i[self.track_i.imageNumber == self.last_im].xBody.iloc[0] 
            initial_x = sub_data.xBody.iloc[0]
            delta = np.abs(final_x - initial_x)
            if delta < size_sum:
                final_y = self.track_i[self.track_i.imageNumber == self.last_im].xBody.iloc[0] 
                initial_y = sub_data.xBody.iloc[0]
                delta = np.abs(final_y - initial_y)
                if delta < size_sum:
                    return True
            return False
        return False

    def potential_fusion(self) -> bool:
        """Check if there is a potential fusion of the two bacteria."""
        if len(self.distance) > 60:
            end_distance = self.distance.distance[-30:].min()
            if end_distance < 20:
                if self.distance.distance[-60: -30].mean() > end_distance:
                    self.last_im = self.distance.im.max()
                    remaining = self.last_disparition()
                    if remaining == "":
                        to_check = self.apparition_to_check()
                        if len(to_check) == 0:
                            return False
                        for id in to_check:
                            if self.check_apparition(id):
                                return True
                        return False
                    else:
                        if self.size_increase(remaining):
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


def distance_analysis_folder(folder: str,
                             res_file: str,
                             pair_distances: pd.DataFrame,
                             tracking_data: pd.DataFrame) -> None:
    """Run the analysis for the folder."""
    pairs = pair_distances.groupby(['i','j']).count().reset_index()[["i", "j"]]
    apparitions = get_apparition(tracking_data)
    for pair in pairs.iterrows():
                ana = DistanceAnalyser(folder, pair_distances, tracking_data, apparitions, pair[1].i, pair[1].j)
                if ana.process():
                    ana.plot_distance()
                    with open(res_file, "a") as file:
                        f = folder.split("/")[-1]
                        file.write(f"{f},{ana.i}, {ana.j},{ana.last_im},0\n")

def main(parent_folder: str) -> None:
    folder_list: List[str] = [os.path.join(parent_folder, f) for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder,f))]
    folder_list.sort()

    log_file = os.path.join(parent_folder, "log.txt")
    with open(log_file, "w") as file:
        file.write("Distance code \n")
    res_file = os.path.join(parent_folder, "Potential_fusion.csv")
    with open(res_file, "w") as file:
        file.write("folder,i,j,last_im,checked\n")
    for folder in folder_list:
        print(folder.split("/")[-1])
        try:
            os.makedirs(os.path.join(folder, "Figure/Distance"))
        except FileExistsError:
            pass
        try:
            try:
                tracking_data = load_data(folder, 30)
                tracking_data = clean(tracking_data)
                pair_distances = pd.read_csv(os.path.join(folder, "Tracking_Result/distances.csv"))
            except (FileNotFoundError, OSError):
                calculator = DistanceCalculator(folder)
                calculator.distance_bacteria()
                pair_distances = calculator.pair_distances
                tracking_data = calculator.data
            except pd.errors.EmptyDataError:
                pair_distances = pd.DataFrame()
            distance_analysis_folder(folder, res_file, pair_distances, tracking_data)
        except KeyError as e:
            pass
        with open(log_file, 'a') as file:
            f = folder.split("/")[-1]
            file.write(f"{f} done at {datetime.now()}\n")    

if __name__=="__main__":
    parent_folder = "/Users/sintes/Desktop/NASGuillaume/Chains/Chains 12%"
    main(parent_folder)