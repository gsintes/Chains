"""Detector based on AI (StarDist) for bacteria detection."""

import os
import json
from typing import List, Tuple

from stardist.models import StarDist2D, Config2D
from stardist import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import numpy as np
import cv2

import base_detector

class AIDetector(base_detector.BaseDetector):
    """Detector based on AI (StarDist) for bacteria detection."""
    def __init__(self, config_folder: str, save_folder: str, visualisation: bool=False) -> None:
        """Initialize the detector.

        Parameters
        ----------
        config_folder : str
            Folder where the configuration files are stored.
        save_folder : str
            Folder where to save the image results.
        visualisation : bool, optional
            Display the image results, by default False

        """
        config_str = json.load(open(f"{config_folder}/config.json", "r"))

        config = Config2D(config_str["axes"],
                          config_str["n_rays"],
                          config_str["n_channel_in"],
                          config_str["grid"],
                          config_str["n_classes"],
                          config_str["backbone"])

        self.model = StarDist2D(basedir="./tmp/")
        self.model.load_weights(f"{config_folder}/weights_best.h5")
        self.model.config = config
        self.model.thresholds = json.load(open(f"{config_folder}/thresholds.json", "r"))
        self.count = 0

        self.save_folder = save_folder
        self.visualisation = visualisation

    def detect(self, image: np.ndarray) -> List[Tuple[np.ndarray,Tuple[int, int]]]:
        """Detect objects

        Parameters
        ----------
        image : ndarray
            Image as GRAYSCALE.

        Returns
        -------
        list
            List of masks as [(mask, left_corner), ...].

        """
        img = np.copy(image)
        labels, _ = self.model.predict_instances(normalize(img))

        if self.visualisation:
            cv2.imwrite(os.path.join(self.save_folder, f"processed{self.count:06d}.png"),
                        render_label(labels, img=img))

        self.count += 1
        return [(labels, (0, 0))]