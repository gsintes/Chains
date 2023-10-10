"""Tracker tools for assignement of object between current and previous objects."""

from typing import List, Dict, Tuple, Any
import os

import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import matplotlib.colors as mcolors

from base_detector import BaseDetector

class Tracker():
    """Tracker class to determine assignment from previous and current coordinates."""

    def __init__(self, params: Dict[str, int] = {}, save_folder: str = "", visualization: bool = True) -> None:
        """Initialize the tracker.

        Parameters
        ----------
        params : dict
            Parameters.
        detector : BaseDetector
            Detector that is an implementation of the BaseDetector.

        """
        if params:
            self.params = params.copy()
        self.is_init = False
        self.save_folder = save_folder
        self.count = 0
        self.visualization = visualization

    def set_params(self, params: Dict[str, int]) -> None:
        """Set the parameters.

        Parameters
        ----------
        params : dict
            Parameters.

        """
        self.params = params.copy()
        self.is_init = False

    def set_detector(self, detector: BaseDetector) -> None:
        """Set the detector.

        Parameters
        ----------
        detector : BaseDetector
            Detector that is an implementation of the BaseDetector.

        """
        self.detector = detector
        self.is_init = False

    def initialize(self, image: np.ndarray) -> List[Dict[str, Dict[str, Any]]]:
        """Initialize the tracker.

        Parameters
        ----------
        image : ndarray
            Image, channels depending on the detector.

        Returns
        -------
        list
            List of detected objects as dict.

        """
        if self.params and self.detector:
            self.prev_detection = self.detector.process(image)
            self.is_init = True
            self.max_id = len(self.prev_detection)
            self.id = list(range(self.max_id))
            self.lost = [0] * len(self.prev_detection)
            self.im = 0
            for i, j in enumerate(self.prev_detection):
                j["3"]["time"] = self.im
                j["3"]["id"] = self.id[i]
            self.im += 1
            return self.prev_detection
        return []

    def process(self, image: np.ndarray) -> List[Dict[str, Dict[str, Any]]]:
        """Process an image.

        Parameters
        ----------
        image : ndarray
            Image, channels depending on the detector.

        Returns
        -------
        list
            List of detected objects as dict.

        """
        if self.is_init:
            self.current_detection = self.detector.process(image)
            order = self.assign(self.prev_detection, self.current_detection)
            losts = self.find_lost(order)
            self.current_detection = self.reassign(self.prev_detection,
                                                   self.current_detection, order)
            while len(self.current_detection) - len(self.id) != 0:
                self.max_id += 1
                self.id.append(self.max_id)
                self.lost.append(0)
            self.current, self.lost, self.id = self.clean(
                self.current_detection, self.lost, losts, self.id)
            for i, j in enumerate(self.current_detection):
                j["3"]["time"] = self.im
                j["3"]["id"] = self.id[i]
            self.im += 1
            self.prev_detection = self.current_detection
            self.lost_ids = losts
            if self.visualization:
                self.make_verif_image(image)
            self.count += 1
            
            return [j for i, j in enumerate(self.current_detection) if i not in losts]
        return []

    @staticmethod
    def angle_difference(a: float, b: float) -> float:
        """Get the minimal difference, a-b), between two angles.

        Parameters
        ----------
        a : float
            First angle.
        b : float
            Second angle.

        Returns
        -------
        float
            a-b.

        """
        a = BaseDetector.modulo(a)
        b = BaseDetector.modulo(b)
        return -(BaseDetector.modulo(a - b + np.pi) - np.pi)

    @staticmethod
    def div(a: float, b: float) -> float:
        """Division by zero, a/0=0.

        Parameters
        ----------
        a : float
            Dividend.
        b : float
            Divisor.

        Returns
        -------
        float
            a/b.

        """
        if b != 0:
            return a/b
        else:
            return 0

    def compute_cost(self, var: List[float], norm: List[float]) -> float:
        """Compute the cost.

        Parameters
        ----------
        var : List
            List of variable.
        norm : list
            Normalization coefficient associated to var.

        Returns
        -------
        float
            Cost.

        """
        cost = 0.
        for i, j in zip(var, norm):
            cost += self.div(i, j)
        return cost

    def assign(self,
               prev: List[Dict[str, Dict[str, Any]]],
               current: List[Dict[str, Dict[str, Any]]]) -> List[int]:
        """Find the optimal assignent.

        Parameters
        ----------
        prev : list
            List of dict. Each dict is one object with 4 key "0", "1", "2", "3".
            0,1,2 is the {center, orientation} of the head, tail and body respectively.
            3 is {area, perim} of the object.
        current : list
            List of dict. Each dict is one object with 4 key "0", "1", "2", "3".
            0,1,2 is the {center, orientation} of the head, tail and body respectively.
            3 is {area, perim} of the object.

        Returns
        -------
        list
            Assignment.

        """
        if len(prev) == 0:
            assignment: List[int] = []
        elif len(current) == 0:
            assignment = [-1] * len(prev)
        else:
            cost = np.zeros((len(prev), len(current)))
            valid = []
            for i, l in enumerate(prev):
                prev_coord = l[str(int(self.params["spot"]))]
                for j, k in enumerate(current):
                    current_coord = k[str(int(self.params["spot"]))]

                    distance = np.sqrt((prev_coord["center"][0] - current_coord["center"][0]) ** 2 + (
                        prev_coord["center"][1] - current_coord["center"][1]) ** 2)
                    angle = np.abs(self.angle_difference(
                        prev_coord["orientation"], current_coord["orientation"]))
                    area = np.abs(l["3"]["area"] - k["3"]["area"])
                    perim = np.abs(l["3"]["perim"] - k["3"]["perim"])

                    if distance < self.params["maxDist"]:
                        cost[i, j] = self.compute_cost([distance, angle, area, perim], [
                                                       self.params["normDist"], self.params["normAngle"], self.params["normArea"], self.params["normPerim"]])
                        valid.append((i, j))
                    else:
                        cost[i, j] = 1e34

            row, col = linear_sum_assignment(cost)

            assignment = []
            for i, _ in enumerate(prev):
                if i in row and (i, col[list(row).index(i)]) in valid:
                    assignment.append(col[list(row).index(i)])
                else:
                    assignment.append(-1)

        return assignment

    def reassign(self,
                 past: List[Dict[str, Dict[str, Any]]],
                 current: List[Dict[str, Dict[str, Any]]],
                 order: List[int]) -> List[Dict[str, Dict[str, Any]]]:
        """Reassign current based on order.

        Parameters
        ----------
        prev : list
            List of dict of previous detections.
        current : list
            List of dict of current detections.
        order : list
            Reassingment

        Returns
        -------
        list
            Reordered current.

        """
        tmp = past
        for i, j in enumerate(past):
            if order[i] != -1:
                tmp[i] = current[order[i]]

        for i, j in enumerate(current):
            if i not in order:
                tmp.append(j)

        return tmp

    def find_lost(self, assignment: List[int]) -> List[int]:
        """Find object lost at previous step.

        Parameters
        ----------
        assignment : list
            Assignment indexes.

        Returns
        -------
        list
            Indexes of lost objects.

        """
        return [i for i, j in enumerate(assignment) if j == -1]

    def clean(self,
              current: List[Dict[str, Dict[str, Any]]],
              counter: List[int],
              lost: List[int],
              idty: List[int]) -> Tuple[List[Dict[str, Dict[str, Any]]], List[int], List[int]]:
        """Delete objects that were lost.
        Only counter is copied in this function. Other lists act as pointer.

        Parameters
        ----------
        current : list
            List to clean.
        counter : list
            Counter of losses.
        lost : list
            Lost objects.
        idty : list
            Objects' identity

        Returns
        -------
        list
            Cleaned list.
        list
            Updated counter.

        """
        counter = [j + 1 if i in lost else 0 for i, j in enumerate(counter)]

        to_delete = sorted([i for i in lost if counter[i] >
                           self.params["maxTime"]], reverse=True)
        for i in to_delete:
            current.pop(i)
            counter.pop(i)
            idty.pop(i)
        return current, counter, idty
    
    @staticmethod
    def hex_to_rgb(value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def make_verif_image(self, image: np.ndarray) -> None:
        """Make a image showing the detection."""
        colors = [Tracker.hex_to_rgb(color) for color in mcolors.TABLEAU_COLORS.values()]
        image_to_draw = cv2.merge([image, image, image])
        font = cv2.FONT_HERSHEY_SIMPLEX
        for object in self.current_detection:
            id_obj = object["3"]["id"]
            if id_obj not in self.lost_ids:
                body_features = object["2"]
                color = colors[id_obj % len(colors)]
                center = (int(body_features["center"][0]), int(body_features["center"][1]))
                center_writing = (int(body_features["center"][0]),
                                int(body_features["center"][1] + body_features["major_axis"]))
                image_to_draw = cv2.ellipse(img=image_to_draw,
                                    center=center,
                                    axes=(int(body_features["major_axis"]), int(body_features["minor_axis"])),
                                    angle=180 * (1 - body_features["orientation"] / np.pi), startAngle=0, endAngle=360,
                                    color=color, thickness=2)
                image_to_draw = cv2.putText(image_to_draw, str(id_obj), org=center_writing, fontFace=font, fontScale=1, color=color)
        cv2.imwrite(os.path.join(self.save_folder, f"tracked{self.count:06d}.png"), image_to_draw)