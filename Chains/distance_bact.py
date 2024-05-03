"""Calculate the distance between two bacteria over time."""

import os
import shutil
from datetime import datetime
import sqlite3
from typing import List, Tuple
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

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
    data["Velocity"] = np.sqrt(data["xVel"] ** 2 + data["yVel"] ** 2)
    return data

def load_data(path: str, frame_rate: int) -> pd.DataFrame:
        """Load the data from the database."""
        dbfile = os.path.join(path, "Tracking_Result/tracking.db")
        con = sqlite3.connect(dbfile)
        df = pd.read_sql_query('SELECT xBody, yBody, bodyMajorAxisLength, imageNumber, id FROM tracking', con)
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
            df = clean(df)
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

def clean(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the data by removing to short tracks."""
    ids = data["id"].unique()
    for id in ids:
        length: float = BACTLENGTH * data.loc[data["id"] == id, "Velocity"].mean() 
        vel: float = SCALE * data.loc[data["id"] == id, "Velocity"].mean() / FRAME_RATE #pix/frame
        if vel < 0.2:
            data: pd.DataFrame = data.drop(data[data["id"] == id].index)
        else:
            thresh = length / vel
            len_track = len(data.loc[data["id"] == id])
            if len_track < thresh:
                data = data.drop(data[data["id"] == id].index)
    return data

class DistanceCalculator:
    """Object to calculate distance between pairs of bacteria over time."""
    def __init__(self, path: str) -> None:

        self.path = path
        self.data = load_data(path, FRAME_RATE)

    def distance_bacteria(self) -> None:
        """Calculate the distance for all pairs of bacteria"""
        ids = self.data.id.unique()
        ids.sort()
        res = []
        pairs = [(i, j) for k, i in enumerate(ids) for j in ids[k + 1:]]
        pool = mp.Pool(mp.cpu_count() -1)
        result = pool.starmap_async(self.distance_pair, pairs)
        res = []
        for value in result.get():
            if len(value) > 2 * FRAME_RATE:
                res += value
        if res:
            res = np.array(res)
            self.pair_distances = pd.DataFrame({
                "i": res[:, 0],
                "j": res[:, 1],
                "im": res[:, 2],
                "distance": res[:, 3]
            })
            self.pair_distances.dropna(inplace=True)
            self.pair_distances.to_csv(os.path.join(self.path, "Tracking_Result/distances.csv"), index=False)

    def distance_pair(self, i: int, j: int) -> List[Tuple[int, float]]:
        """Calculate the distance for all times """
        data = self.data[self.data["id"].isin((i, j))]
        images = list(data["imageNumber"].unique())
        res = []
        for im in images:
            sub_data = data[data["imageNumber"]==im]
            if len(sub_data) == 2:
                diff = sub_data.diff().dropna()
                distance = np.sqrt(diff.xBody.max() ** 2 + diff.yBody.max() ** 2)
                res.append((i, j, im, distance))
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
        fig_folder = os.path.join(folder, "Figure/Distance")
        try:
            os.makedirs(fig_folder)
        except FileExistsError:
            shutil.rmtree(fig_folder)
            os.makedirs(fig_folder)
        try:
            tracking_data = load_data(folder, 30)
            try:
                pair_distances = pd.read_csv(os.path.join(folder, "Tracking_Result/distances.csv"))
            except (FileNotFoundError, OSError, pd.errors.EmptyDataError):
                calculator = DistanceCalculator(folder)
                calculator.distance_bacteria()
                pair_distances = calculator.pair_distances
                tracking_data = calculator.data
            if not pair_distances.empty:
                distance_analysis_folder(folder, res_file, pair_distances, tracking_data)
        except NotEnoughDataError as e:
            pass
        with open(log_file, 'a') as file:
            f = folder.split("/")[-1]
            file.write(f"{f} done at {datetime.now()}\n")

if __name__=="__main__":
    parent_folder = "/Volumes/Guillaume/Chains/Chains 13.7%"
    main(parent_folder)