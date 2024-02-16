"""Analysis the manual tracking."""

import os
from typing import Dict, Any
import json

import numpy as np
import pandas as pd

def load_data(file: str) -> pd.DataFrame:
    """Load the data from the manual tracking."""
    scale = 6.24
    f = open(file)
    data: Dict[str, Any] = json.load(f)
    coords = data["coords"]
    diffs = []
    for i in range(len(coords) - 1):
        diffs.append(np.sqrt((coords[i + 1][0] - coords[i][0]) ** 2 + (coords[i + 1][1] - coords[i][1]) ** 2))
    mean_vel = np.mean(diffs)
    data.pop("coords")
    data["vel"] = mean_vel / scale
    return pd.DataFrame(data, index=[0])


if __name__=="__main__":
    folder = "/Users/sintes/Desktop/NASGuillaume/Chains/Manual_tracks/"
    list_file = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".txt")]
    data = pd.DataFrame()
    for file in list_file:
        sub_data = load_data(file)
        data = pd.concat([data, sub_data], ignore_index=True)

