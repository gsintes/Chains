"""Detector for chain detection."""

import os
from typing import List, Tuple, Dict

import numpy as np
import cv2

import preprocessing
from base_detector import BaseDetector

class ChainDetector(BaseDetector):
    """Implement the chain detector. """
    def __init__(self,
                 params: Dict[str, int],
                 save_folder: str,
                 visualisation: bool = False) -> None:
        """Initialize the detector.

        Parameters
        ----------
        params : dict
            Parameters.

        """
        self.params = params
        self.save_folder = save_folder
        self.visualisation = visualisation
        self.count = 0
        self.kernel_size = 7
        self.nb_iter = 2

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
     
        if int(self.params["lightBack"]) == 0:
            image = self.background - image
        else:
            image = image - self.background

        image = preprocessing.binarize(image)
        image = preprocessing.remove_small_objects(image)
        image = preprocessing.elongate_objects(image, self.nb_iter, self.kernel_size)
      
        if int(self.params["xBottom"]) != 0 and int(self.params["yBottom"]) != 0:
            image = image[int(self.params["yTop"]):int(self.params["yBottom"]), int(
                self.params["xTop"]):int(self.params["xBottom"])]
            
        image = image * 255
        image = image.astype(np.uint8)

        if self.visualisation:
            cv2.imwrite(os.path.join(self.save_folder, f"processed{self.count:06d}.png"), image)

        contours, _ = cv2.findContours(
            image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        masks: List[Tuple[np.ndarray, Tuple[int, int]]] = []
        for i in contours:
            area = cv2.contourArea(i)
            if area < int(self.params["maxArea"]) and area > int(self.params["minArea"]):
                rect = cv2.boundingRect(i)
                mask = np.zeros_like(image)
                cv2.drawContours(mask, [i], 0, 255, -1, 8)
                masks.append(
                    (np.copy(mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]), rect[0:2]))
        self.count += 1

        return masks

    def set_background(self, image: np.ndarray) -> None:
        """Set the background image.

        Parameters
        ----------
        image : ndarray
            Background image.

        """
        self.background = np.copy(image)
