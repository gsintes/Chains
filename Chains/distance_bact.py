"""Calculate the distance between two bacteria over time."""

import os
import shutil
from datetime import datetime
import sqlite3
from typing import List, Tuple
import multiprocessing as mp
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

from tracking_analysis import Analysis

import time

pd.options.mode.chained_assignment = None  # default='warn'
BACTLENGTH = 10
FRAME_RATE = 30
SCALE = 6.24

class NotEnoughDataError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def calculate_velocity(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate velocities."""
    ids = data["id"].unique()
    for id in ids:
        sub_data = data.loc[data["id"] == id]
        coord = ["xBody", "yBody"]
        for ax in coord:
            pos_diff = sub_data[ax].diff() / SCALE
            time_diff = sub_data["time"].diff()
            velocity = pos_diff / time_diff
            data.loc[velocity.index, ax[0] + "Vel"] = velocity
    data["velocity"] = np.sqrt(data["xVel"] ** 2 + data["yVel"] ** 2)
    return data

def load_data(path: str, frame_rate: int) -> pd.DataFrame:
        """Load the data from the database."""
        dbfile = os.path.join(path, "Tracking_Result/tracking.db")
        con = sqlite3.connect(dbfile)
        df = pd.read_sql_query('SELECT xBody, yBody, tBody, bodyMajorAxisLength, bodyMinorAxisLength, imageNumber, id FROM tracking', con)
        con.close()
        df["time"] = df["imageNumber"] / frame_rate

        df["id"] = pd.to_numeric(df["id"], downcast="unsigned")
        df["imageNumber"] = pd.to_numeric(df["imageNumber"], downcast="signed")
        df["xBody"] = pd.to_numeric(df["xBody"], downcast="float")
        df["yBody"] = pd.to_numeric(df["yBody"], downcast="float")
        df["time"] = pd.to_numeric(df["time"], downcast="float")
        df["bodyMajorAxisLength"] = pd.to_numeric(df["bodyMajorAxisLength"], downcast="float")

        if len(df.id.unique()) > 1:
            df = calculate_velocity(df)
            df = Analysis.clean(df, SCALE, FRAME_RATE)
            if len(df.id.unique()) > 1:
                return df
        raise NotEnoughDataError("Need at least two differents chains.")

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



class DistanceCalculator:
    """Object to calculate distance between pairs of bacteria over time."""
    def __init__(self, path: str) -> None:
        self.path = path
        self.data = load_data(path, FRAME_RATE)

    def distance_bacteria(self) -> None:
        """Calculate the distance for all pairs of bacteria"""
        ids = self.data.id.unique()
        ids.sort()
        self.save_file = os.path.join(self.path, "Tracking_Result/distances.csv")
        with open(self.save_file, "w") as f:
            f.write("i,j,im,distance\n")
        grouped = self.data.groupby("imageNumber")["id"].apply(lambda x:
                                                               list(combinations(x.values,2))).apply(pd.Series).stack()\
                                                                .reset_index(level=0,name='ids')
        v_counts = grouped.ids.value_counts()
        pairs = list(v_counts[v_counts > 2 * FRAME_RATE].index)

        pool = mp.Pool(mp.cpu_count() -1)
        _ = pool.starmap(self.distance_pair, pairs)

    def distance_pair(self, i: int, j: int) -> None:
        """Calculate the distance for all times."""
        data = self.data[self.data["id"].isin((i, j))]
        grouped = data.groupby("imageNumber")

        grouped_data = pd.DataFrame()
        grouped_data["nb"] = grouped.imageNumber.value_counts()
        grouped_data = grouped_data[grouped_data.nb==2]
        images = list(grouped_data.index)
        images.sort(reverse=True)
        res = []

        for im in images[0:60]:
            sub_data = data[data["imageNumber"]==im]
            diff = sub_data.diff().dropna()
            distance = np.sqrt(diff.xBody.max() ** 2 + diff.yBody.max() ** 2)

            res.append((i, j, im, distance))
        with open(self.save_file, "a") as f:
            for r in res:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.1f}\n")

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
            if delta < 0.5 * size_sum:
                final_y = self.track_i[self.track_i.imageNumber == self.last_im].xBody.iloc[0] 
                initial_y = sub_data.xBody.iloc[0]
                delta = np.abs(final_y - initial_y)
                if delta < 0.5 * size_sum:
                    return True
            return False
        return False
    
    def is_decreasing(self) -> bool:
        """Check if the distance is decreasing in the last part."""
        last_seconds = self.distance[self.distance.im >=  (self.last_im - 2 * FRAME_RATE)]
        if len(last_seconds) < 30:
            return False
        x = last_seconds[["im"]]
        y = last_seconds["distance"]
        self.model = LinearRegression()
        self.model.fit(x, y)
        if self.model.score(x, y) > 0.7:
            if self.model.coef_ < 0:
                self.x = x
                return True
        return False

    def potential_fusion(self) -> bool:
        """Check if there is a potential fusion of the two bacteria."""
        if len(self.distance) > 2 * FRAME_RATE:
            self.last_im = self.distance.im.max()
            end_distance = self.distance[self.distance.im >=  (self.last_im - FRAME_RATE)].distance.min()
            if end_distance < 20:
                if self.is_decreasing(): 
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
        plt.plot(self.x, self.model.predict(self.x))
        plt.ylim(bottom=0)
        plt.xlabel("Image")
        plt.ylabel("Distance (pixel)")
        plt.title(f"Pair: ({self.i}, {self.j})")
        file_name = os.path.join(self.path, f"Figure/Distance/{int(self.i)}-{int(self.j)}.png")
        plt.savefig(file_name)
        plt.close()

def task(folder, res_file, pair_distances, tracking_data, apparitions, i, j):
    ana = DistanceAnalyser(folder, pair_distances, tracking_data, apparitions, i, j)
    if ana.process():
        with open(res_file, "a") as file:
            f = folder.split("/")[-1]
            file.write(f"{f},{int(ana.i)},{int(ana.j)},{int(ana.last_im)},0\n")
                        
def distance_analysis_folder(folder: str,
                             res_file: str,
                             pair_distances: pd.DataFrame,
                             tracking_data: pd.DataFrame) -> None:
    """Run the analysis for the folder."""
    pairs = pair_distances.groupby(['i','j']).count().reset_index()[["i", "j"]]
    apparitions = get_apparition(tracking_data)
    args = [(folder, res_file, pair_distances, tracking_data, apparitions, pair[1].i, pair[1].j) for pair in pairs.iterrows()]
    pool = mp.Pool(mp.cpu_count() -1)
    pool.starmap_async(task, args)

def read_pair_distances(folder: str) -> pd.DataFrame:
    """Read the pair distances and reduce the type of data to reduce memory usage."""
    data = pd.read_csv(os.path.join(folder, "Tracking_Result/distances.csv"))
    data["i"] = pd.to_numeric(data["i"], downcast="unsigned")
    data["j"] = pd.to_numeric(data["j"], downcast="unsigned")
    data["im"] = pd.to_numeric(data["im"], downcast="signed")
    data["distance"] = pd.to_numeric(data["distance"], downcast="float")
    return data

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
        fig_folder = os.path.join(folder, "Figure/Distance")
        try:
            os.makedirs(fig_folder)
        except FileExistsError:
            shutil.rmtree(fig_folder)
            os.makedirs(fig_folder)
        try:
            tracking_data = load_data(folder, 30)
            try:
                pair_distances = read_pair_distances(folder)
                print("Read data distance")
            except (FileNotFoundError, OSError, pd.errors.EmptyDataError):
                print("Calculating distance")
                calculator = DistanceCalculator(folder)
                calculator.distance_bacteria()
                pair_distances = read_pair_distances(folder)
                tracking_data = calculator.data
                print("Finished calculation")
            if not pair_distances.empty:
                distance_analysis_folder(folder, res_file, pair_distances, tracking_data)
                print("Fusion detection finished")
        except NotEnoughDataError as e:
            pass
        with open(log_file, 'a') as file:
            f = folder.split("/")[-1]
            file.write(f"{f} done at {datetime.now()}\n")

if __name__=="__main__":
    # parent_folder = "/Volumes/Guillaume/ChainFormation"
    # main(parent_folder)

    folder = "/Users/sintes/Desktop/Test"
    calc = DistanceCalculator(folder)
    calc.distance_bacteria()
