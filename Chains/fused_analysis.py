"""Take the data from all experiments and fuse them."""

import os
import re

import pandas as pd

def get_concentration(concentration_folder: str) -> float:
    """Get the concentration from the folder name."""
    folder_name = concentration_folder.split("/")[-1]
    template = r"Chains\ (\d+.?\d*)\%"
    group = re.match(template, folder_name)
    return float(group[1])

def load_all_data(parent_folder: str) -> pd.DataFrame:
    """Load all the datas from the subfolders."""
    concentration_folders = [os.path.join(parent_folder, f) for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder,f))]
    data = pd.DataFrame()
    for c_folder in concentration_folders:
        concentration = get_concentration(c_folder)
        sub_folders = [os.path.join(c_folder, f) for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(c_folder,f))]
        for f in sub_folders:
            sub_data = pd.read_csv(os.path.join(f, "Tracking_Result/vel_data.csv"))
            sub_data["Concentration_LC"] = concentration
            data = pd.concat(data, sub_data)
if __name__ == "__main__":
    parent_folder = "/Volumes/Chains/Chains"
    data = load_all_data(parent_folder)

    
