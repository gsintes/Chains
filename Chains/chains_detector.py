"""Detector for chain detection."""

from base_detector import BaseDetector
import numpy as np
import cv2

import preprocessing


class ChainDetector(BaseDetector):
    """Implement the chain detector.

    """

    def __init__(self, params):
        """Initialize the detector.

        Parameters
        ----------
        params : dict
            Parameters.

        """
        self.params = params

    def detect(self, image):
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
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        print(np.amin(image), np.amax(image))
        if int(self.params["lightBack"]) == 0:
            image = self.background - image
        else:
            image = image - self.background
        print(np.amin(image), np.amax(image))
        cv2.imshow("subtracted", image)
        cv2.waitKey(0)
        __, image = cv2.threshold(image, int(
            self.params["thresh"]), 255, cv2.THRESH_BINARY)

        image = preprocessing.remove_small_objects(image)
        image = preprocessing.elongate_objects(image)

        if int(self.params["xBottom"]) != 0 and int(self.params["yBottom"]) != 0:
            image = image[int(self.params["yTop"]):int(self.params["yBottom"]), int(
                self.params["xTop"]):int(self.params["xBottom"])]

        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        masks = []
        for i in contours:
            area = cv2.contourArea(i)
            if area < int(self.params["maxArea"]) and area > int(self.params["minArea"]):
                rect = cv2.boundingRect(i)
                mask = np.zeros_like(image)
                cv2.drawContours(mask, [i], 0, 255, -1, 8)
                masks.append(
                    (np.copy(mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]), rect[0:2]))

        return masks

    def set_background(self, image):
        """Set the background image.

        Parameters
        ----------
        image : ndarray
            Background image.

        """
        self.background = np.copy(image)
