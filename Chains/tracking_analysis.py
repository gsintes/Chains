"""Analysis of the tracking data."""

import os
from typing import List
import json

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from chains_detector import ChainDetector

np.seterr(all='raise')

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class Analysis():
    """Class to perform the analysis of chain tracking data."""
    def __init__(self, path: str) -> None:
        self.path = path
        self.bactLength = 10
        self.frameRate = self.read_frame_rate()
        self.scale = 6.24
        self.data_ind = self.load_data()

    def read_frame_rate(self) -> int:
        """Read the frame rate."""
        info_file = os.path.join(self.path, "RawImageInfo.txt")
        file = open(info_file, "r")
        line = file.readline()
        return int(line)

    def load_data(self) -> pd.DataFrame:
        """Load the data from the database."""
        dbfile = os.path.join(self.path, "Tracking_Result/tracking.db")
        con = sqlite3.connect(dbfile)
        df = pd.read_sql_query('SELECT xBody, yBody, bodyMajorAxisLength, bodyMinorAxisLength, tBody, imageNumber, id FROM tracking', con)
        con.close()

        df["time"] = df["imageNumber"] / self.frameRate

        df["id"] = pd.to_numeric(df["id"], downcast="unsigned")
        df["imageNumber"] = pd.to_numeric(df["imageNumber"], downcast="signed")
        df["xBody"] = pd.to_numeric(df["xBody"], downcast="float")
        df["tBody"] = pd.to_numeric(df["tBody"], downcast="float")
        df["yBody"] = pd.to_numeric(df["yBody"], downcast="float")
        df["time"] = pd.to_numeric(df["time"], downcast="float")
        df["bodyMajorAxisLength"] = pd.to_numeric(df["bodyMajorAxisLength"], downcast="float")

        return df

    @staticmethod
    def calculate_velocity(data: pd.DataFrame, scale: float) -> pd.DataFrame:
        """Calculate velocities."""
        ids = data["id"].unique()
        ids.sort()
        for id in ids:
            sub_data = data.loc[data["id"] == id]
            coord = ["xBody", "yBody"]
            for ax in coord:
                pos_diff = sub_data[ax].diff() / scale
                time_diff = sub_data["time"].diff()

                velocity = pos_diff / time_diff
                data.loc[velocity.index, ax[0] + "Vel"] = velocity
        data["velocity"] = np.sqrt(data["xVel"] ** 2 + data["yVel"] ** 2)
        return data


    @staticmethod
    def clean(data: pd.DataFrame, scale: float, frame_rate: int, bactLength: int=10) -> pd.DataFrame:
        """Clean the data by removing to short tracks."""
        data.dropna(inplace=True)
        grouped = data.groupby("id")
        vel = scale * grouped["velocity"].mean() / frame_rate
        duration = grouped["imageNumber"].max() - grouped["imageNumber"].min()
        count = data.id.value_counts()
        data = data.set_index("id")
        data["v"] = vel
        data["l"] = count
        data["prop"] = count / duration
        data = data.reset_index()
        data = data.drop(data[(data.v < 0.2) |
                                             (data.l < bactLength * frame_rate / scale)
                                             | (data.prop < 0.75)].index)
        data = data.drop(["v", "l"], axis=1)
        return data

    def trace_image(self) -> None:
        """Draw the image with superimposed trajectories."""
        plt.figure()
        plt.xlim((0, 1024))
        plt.ylim((0, 1024))
        x = np.linspace(0, 1024)
        y = self.main_axis[1] * (x - 512) / self.main_axis[0] + 512
        ids = self.chain_data["id"].unique()
        for id in ids:
            sub_data: pd.DataFrame = self.chain_data.loc[self.chain_data["id"]==id]
            plt.plot(sub_data["xBody"], 1024 - sub_data["yBody"], ".", markersize=1)
        plt.plot(x, y, "k--")
        plt.savefig(os.path.join(self.path, "Figure/trace.png"))
        plt.close()

    def main_direction(self) -> None:
        """Detect the main direction of the swimming.""" #TODO: rewrite this function
        x = np.array(self.data_ind["xBody"])
        y = np.array(self.data_ind["yBody"])
        cov = np.cov(x, y)
        val, vect = np.linalg.eig(cov)
        val = list(val)
        i = val.index(max(val))
        self.main_axis = vect[:, i]

    def sign_trajectory(self, id) -> int:
        """Get a sign of the trajectory.
        Return 1 if swimming one way, -1 otherwise. Sign arbitrary."""
        subdata: pd.DataFrame = self.chain_data.loc[self.chain_data["id"]==id]
        mean_xvel = subdata["xVel"].mean()
        mean_yvel = subdata["yVel"].mean()
        ps = mean_xvel * self.main_axis[0] + mean_yvel * self.main_axis[1]
        return np.sign(ps)

    def get_chain_position(self) -> None:
        """Get the chain position."""
        ids = self.chain_dict.keys()
        for id in ids:
            bacts = self.chain_dict[id]["bacts"]
            steps = self.chain_dict[id]["steps"]
            sub_ind_data = self.data_ind[self.data_ind["id"].isin(bacts)]
            sub_ind_data = sub_ind_data[sub_ind_data["imageNumber"].isin(steps)]
            x_bodys = sub_ind_data.groupby("imageNumber")["xBody"].mean()
            x_bodys.sort_index(inplace=True)
            y_bodys = sub_ind_data.groupby("imageNumber")["yBody"].mean()
            y_bodys.sort_index(inplace=True)
            sub_chain_data = self.chain_data[self.chain_data["id"] == id]

            try:

                self.chain_data.loc[sub_chain_data.index, "xBody"] = x_bodys.values
                self.chain_data.loc[sub_chain_data.index, "yBody"] = y_bodys.values
            except ValueError:
                print(id, len(sub_chain_data), len(x_bodys))

    def process(self) -> pd.DataFrame:
        """Performs the analysis."""
        self.data_ind = Analysis.calculate_velocity(self.data_ind, self.scale)
        self.data_ind = Analysis.clean(self.data_ind, self.scale, self.frameRate)
        c_detector = ChainDetector(self.data_ind)
        self.chain_data, self.chain_dict = c_detector.process()
        self.chain_data["time"] = self.chain_data["imageNumber"] / self.frameRate
        self.get_chain_position()
        self.main_direction()
        self.trace_image()
        self.chain_data = self.calculate_velocity(self.chain_data, self.scale)

        ids = self.chain_data["id"].unique()
        chain_length: List[int] = []
        mean_vel: List[float] = []
        signs: List[int] = []
        for id in ids:
            data: pd.DataFrame = self.chain_data.loc[self.chain_data["id"] == id]
            chain_length.append(int(min(data["chain_length"])))
            mean_vel.append(data["velocity"].mean())
            signs.append(self.sign_trajectory(id))
        self.velocity_data = pd.DataFrame({"id": ids,
                             "chain_length": chain_length,
                             "velocity": mean_vel,
                             "sign": signs})

        single_vel = self.velocity_data.loc[self.velocity_data["chain_length"] == 1, "velocity"].mean()
        self.velocity_data["Single_vel"] = single_vel
        self.velocity_data["Normalized_vel"] = self.velocity_data["velocity"] / single_vel
        self.chain_data.to_csv(os.path.join(self.path, "Tracking_Result/chain_data.csv"), index=None)
        json.dump(self.chain_dict, open(os.path.join(self.path, "Tracking_Result/chain_dict.json"), "w"), cls=NpEncoder)

        return self.velocity_data

if __name__ == "__main__":
    # parent_folder = "/Volumes/Chains/ChainFormation/"
    # folder_list: List[str] = [os.path.join(parent_folder, f) for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder,f))]
    # folder_list.sort()
    # for folder in folder_list[0:1]:
    folder = "/Users/sintes/Desktop/ImageSeq"
    print(folder)
    analysis = Analysis(folder)
    if len(analysis.data_ind) > 0:
        velocity_data = analysis.process()
        velocity_data.to_csv(os.path.join(folder,"Tracking_Result/vel_data.csv"), index=None)
