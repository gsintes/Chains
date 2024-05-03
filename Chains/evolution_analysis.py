"""Analyse the evolution of a sample with time."""

import os
from typing import Tuple

import pandas as pd

from tracking_analysis import Analysis

folder = "/Volumes/Guillaume"

import time

class FolderAnalysis(Analysis):
    """Analyse the folder by time interval."""
    time_interval = 300 #5min in s

    def __init__(self, folder: str) -> None:
        super().__init__(folder)
        self.time_duration = self.data.time.max()
        self.nb_time_inter = int(1 + self.time_duration // self.time_interval) 
 
    def clean(self) -> None:
        """Clean the data"""
        pass #TODO implement

    def group_by_time(self, time_begining: int, time_end: int) -> pd.DataFrame:
        """Group the data by time interval."""
        b = time.time()
        data = self.data[(self.data.time >= time_begining) & (self.data.time < time_end)]
        lengths = data.chain_length.unique()

        grouped_df = pd.DataFrame({}, index=lengths)
        grouped = data.groupby("chain_length")
        
        grouped_df["nb_chains"] = grouped.id.nunique()
        grouped_df[["mean_vel", "std_vel", "min_vel", "max_vel"]] = grouped.Velocity.agg(["mean", "std", "min", "max"])
        
        grouped_df["begining_time"] = time_begining
        grouped_df["end_time"] = time_end
        grouped_df["mean_time"] = (time_begining + time_end) / 2

        return grouped_df
        
    def __call__(self) -> None:
        """Process the analysis."""
        self.calculate_velocity()
        self.calculate_chain_length()
        
        self.data = self.data.drop(["xBody", "yBody", "bodyMajorAxisLength"], axis=1)
        self.clean()
        self.group_by_time(0, 300)

        

if __name__=="__main__":
    ana = FolderAnalysis("/Volumes/Guillaume/ChainFormation/2024-03-12_18h37m34s")
    ana()