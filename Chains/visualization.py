"""Redo the visualization of the tracking on the images."""

import os
import shutil
import json
from typing import List

import cv2
import numpy as np
import matplotlib.colors as mcolors

from tracker import Tracker
from preprocessing import convert_16to8bits
from distance_bact import load_data, NotEnoughDataError


class Visualisation:
    """"Perform the analysis """
    def __init__(self, folder) -> None:
        self.folder = folder
        self.fig_folder = os.path.join(self.folder, "Figure/Tracked")
        try:
            os.makedirs(self.fig_folder)
        except FileExistsError:
            shutil.rmtree(self.fig_folder)
            os.makedirs(self.fig_folder)
        self.data = load_data(folder, 30)
        self.data
        self.image_list = [os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith(".tif")]
        self.image_list.sort()
        param_file = os.path.join(self.folder, "params.json")
        f = open(param_file)
        parsed_params = json.load(f)
        self.max_int = parsed_params["maxint"]

    def process(self) -> None:
        """Run the verif image for all step image in the folder."""
        for i in range(len(self.image_list)):
            self.make_verif_image(i)

    def make_verif_image(self, i: int) -> None:
            """Make a image showing the detection."""
            colors = [Tracker.hex_to_rgb(color) for color in mcolors.TABLEAU_COLORS.values()]
            image = convert_16to8bits(self.image_list[i], self.max_int)
            image_to_draw = cv2.merge([image, image, image])
            font = cv2.FONT_HERSHEY_SIMPLEX

            sub_data = self.data[self.data.imageNumber == i]
            for _, row in sub_data.iterrows():
                id_obj = int(row.id)
                color = colors[id_obj % len(colors)]
                center = (int(row.xBody), int(row.yBody))
                center_writing = (int(row.xBody),
                                int(row.yBody + row.bodyMajorAxisLength))
                image_to_draw = cv2.ellipse(img=image_to_draw,
                                    center=center,
                                    axes=(int(row.bodyMajorAxisLength), int(row.bodyMinorAxisLength)),
                                    angle=180 * (1 - row.tBody / np.pi), startAngle=0, endAngle=360,
                                    color=color, thickness=1)
                image_to_draw = cv2.putText(image_to_draw, str(id_obj), org=center_writing, fontFace=font, fontScale=1, color=color)
            cv2.imwrite(os.path.join(self.fig_folder, f"tracked{i:06d}.png"), image_to_draw)

if __name__=="__main__":
    parent_folder = "/Users/sintes/Desktop/NASGuillaume/Chains/Chains 11%"
    folder_list: List[str] = [os.path.join(parent_folder, f) for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder,f))]
    for folder in folder_list:
        try:
            vis = Visualisation(folder)
            vis.process()
        except NotEnoughDataError:
            pass