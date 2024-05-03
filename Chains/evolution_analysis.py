"""Analyse the evolution of a sample with time."""

import os
import sqlite3

import pandas as pd

from tracking_analysis import Analysis

folder = "/Volumes/Guillaume"

class FolderAnalysis(Analysis):
    """Analyse the folder by time interval."""
    time_interval = 300 #5min in s

    def __init__(self, folder: str) -> None:
        super().__init__(folder)
        self.time_duration = self.data.time.max()
        self.nb_time_inter = int(1 + self.time_duration // self.time_interval)
        print(self.nb_time_inter)

    def group_by_time(self) -> None:
        """Group the data by time interval."""
        pass

    def __call__(self) -> None:
        self.calculate_velocity()
        self.calculate_chain_length()
        

if __name__=="__main__":
    ana = FolderAnalysis("/Volumes/Guillaume/ChainFormation/2024-03-12_18h37m34s")
    ana()