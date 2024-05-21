import abc
from typing import List, Tuple, Dict, Union

import numpy as np
import cv2


class BaseDetector(metaclass=abc.ABCMeta):
    """Abstract class to implement an objects detector."""

    @abc.abstractmethod
    def detect(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """Abstract method to be implemented.

        This method will take a full image with all the objects to detect and will return
        a list of tuples (mask, left_corner_coordinate [x, y]) with one object by mask, the object represented by non-zero pixels
        and the background by zero pixels.


        Parameters
        ----------
        image : ndarray
            The full image.

        Returns
        -------
        dict
            List of (mask, left_corner_coord).

        """
        pass

    def process(self, image: np.ndarray) -> List[Dict[str, Union[float, int]]]:
        """Process one image.

        Parameters
        ----------
        image : ndarray
            The full image.

        Returns
        -------
        list
            List of detected objects and their features.

        """
        detections: List[Dict[str, Union[float, int]]] = []
        for mask, coordinate in self.detect(image):
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            body = self.get_features(mask)
            body["area"] = cv2.contourArea(
                contours[0])
            body["perim"] = cv2.arcLength(contours[0], True)

            body["xcenter"] += coordinate[0]
            body["ycenter"] += coordinate[1]
            detections.append(body)
        return detections

    def get_features(self, mask: np.ndarray) -> Dict[str, Union[float, int]]:
        """Get the object features using equivalent ellipse.


        Parameters
        ----------
        mask : ndarray
            Mask of one object.

        """
        moments = cv2.moments(mask)

        x = moments["m10"] / moments["m00"]
        y = moments["m01"] / moments["m00"]

        i = moments["mu20"]
        j = moments["mu11"]
        k = moments["mu02"]
        if i - k != 0:
            orientation = (0.5 * np.arctan((2 * j) / (i - k)) +
                           (i < k) * (np.pi * 0.5))
            orientation += 2 * np.pi * (orientation < 0)
            orientation = (2 * np.pi - orientation)
        else:
            orientation = 0

        maj_axis = 2 * \
            np.sqrt((((i + k) + np.sqrt((i - k) * (i - k) + 4 * j * j))
                    * 0.5) / moments["m00"])
        min_axis = 2 * \
            np.sqrt((((i + k) - np.sqrt((i - k) * (i - k) + 4 * j * j))
                    * 0.5) / moments["m00"])

        return {"xcenter": x, "ycenter": y, "orientation": orientation, "major_axis": maj_axis, "minor_axis": min_axis}

    @staticmethod
    def modulo(angle: float) -> float:
        """Provide the mathematical 2pi modulo.


        Parameters
        ----------
        mask : float
            Angle in radian.

        Returns
        -------
        float
            Angle between 0->2pi

        """
        return angle - 2 * np.pi * np.floor(angle / (2 * np.pi))

    def get_direction(self, mask: np.ndarray, features: Dict[str, Union[float, int]]) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Get the object direction.

        The object orientation is updated with the correct direction.


        Parameters
        ----------
        mask : ndarray
            Mask of one object.
        features : dict
            Object features.

        Returns
        -------
        bool
            Is object left oriented.
        ndarray
            Rotated mask.
        ndarray
            Rotation matrix.

        """
        rot = cv2.getRotationMatrix2D(center=(
            mask.shape[1] / 2, mask.shape[0] / 2), angle=-(features["orientation"] * 180) / np.pi, scale=1)
        new_size = [int(mask.shape[0] * np.abs(rot[0, 1]) + mask.shape[1] * np.abs(rot[0, 0])),
                    int(mask.shape[0] * np.abs(rot[0, 0]) + mask.shape[1] * np.abs(rot[0, 1]))]
        rot[0, 2] += new_size[0] / 2 - mask.shape[1] / 2
        rot[1, 2] += new_size[1] / 2 - mask.shape[0] / 2
        rotated_mask = cv2.warpAffine(mask, rot, new_size)
        dist = np.sum(rotated_mask, axis=0, dtype=np.float64)
        dist /= np.sum(dist)
        indexes = np.arange(1, len(dist)+1, dtype=np.float64)
        mean = np.sum(indexes*dist)
        sd = np.sqrt(np.sum((indexes-mean)**2*dist))
        skew = (np.sum(indexes ** 3 * dist) -
                3 * mean * sd ** 2 - mean ** 3) / sd ** 3
        if skew > 0:
            features["orientation"] = self.modulo(
                features["orientation"] - np.pi)
            return True, rotated_mask, rot
        else:
            return False, rotated_mask, rot
