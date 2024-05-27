"""Detector based on AI (StarDist) for bacteria detection."""

import json
from typing import List, Tuple

from stardist.models import StarDist2D, Config2D
from csbdeep.utils import normalize
import numpy as np
import cv2

from base_detector import BaseDetector

class AIDetector(BaseDetector):
    """Detector based on AI (StarDist) for bacteria detection."""
    def __init__(self, config_folder: str) -> None:
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
        labels, infos = self.model.predict_instances(normalize(img))
        masks = []
        for i in range(infos["points"].shape[0]):
            center = infos["points"][i]
            value = labels[center[0],  center[1]]
            mask = np.zeros_like(labels, dtype=np.uint8)
            mask[labels == value] = 255
            contours, _ = cv2.findContours(
                image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            rect = cv2.boundingRect(contours[0])
            masks.append((np.copy(mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]), rect[0:2]))
        self.count += 1
        return masks