"""Tracker based on Kalman filters."""
import os
from typing import List, Dict, Union

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import matplotlib.colors as mcolors

from base_detector import BaseDetector


class Particle:
    """A particle is an object tracked in the image."""
    def __init__(self, identifier: int):
        self.is_init = False
        self.id = identifier
        self.attributes: Dict[str, float] = {}

        self._kf = KalmanFilter(4, 2)
        self._kf.F = np.array([[1., 0., 1., 0.],
                              [0., 1., 0., 1.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]])
        self._kf.H = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.]])
        self._kf.P *= 1000. #TODO fix value for the 3 next param
        self._kf.R = [[2, 0], [0, 2]] # Error in estimate
        self._kf.Q = Q_discrete_white_noise(dim=4, dt=0.1, var=0.13)
        self.skip_count = 0

    def init_filter(self, state: np.ndarray) -> None:
        """Initialise the filter."""
        self.is_init = True
        self._kf.x = state

    def predict(self) -> np.ndarray:
        """Predict the position."""
        self._kf.predict()
        return self._kf.x

    def update(self, measure: Union[None, np.ndarray]) -> None:
        """Predict the position."""
        self._kf.update(measure)

    def update_attributes(self, detection: Dict[str, float], time: int) -> None:
        """Update the attributes based on the detection."""
        self.attributes = detection
        self.attributes["time"] = time
        self.attributes["id"] = self.id

class ObjectTracker:
    """Track the objects in a video and solve the assignment issue."""
    def __init__(self, params: Dict[str, int],
                 detector: BaseDetector, save_folder: str = "",
                 visualization: bool = True):
        self.particles: List[Particle] = []
        self.params = params
        self.max_id = 0
        self.is_init = False
        self.save_folder = save_folder
        self.visualization = visualization
        self.detector = detector
        self.im = 0


    def initialize(self, image: np.ndarray) -> List[Dict[str, float]]:
        """Initialize the tracker on the first images.

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
            self.is_init = True
            detection = self.detector.process(image)
            for detec in detection:
                particle = Particle(self.max_id)
                particle.update_attributes(detec, self.im)
                self.particles.append(particle)
                self.max_id += 1
            data = [part.attributes for part in self.particles]
            if self.visualization:
                self.make_verif_image(image, data)
            return data
        return []

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
        return 0

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
            cost += ObjectTracker.div(i, j)
        return cost

    def process(self, image: np.ndarray) -> List[Dict[str, float]]:
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
            current_detection = self.detector.process(image)
            order = self.assign(current_detection)
            losts = self.find_lost(order)
            self.update(current_detection, order)
            self.clean(losts)
            self.im += 1
            tracking_data = [part.attributes for i, part in enumerate(self.particles) if i not in losts]
            if self.visualization:
                self.make_verif_image(image, tracking_data)

            return tracking_data
        return []

    def assign(self, current_detection: List[Dict[str, float]]) -> List[int]:
        """Find the optimal assignent.
        Returns
        -------
        list
            Assignment.
        """
        if not self.particles:
            assignment: List[int] = []
        elif not current_detection:
            assignment = [-1] * len(self.particles)
        else:
            cost = np.zeros((len(self.particles), len(current_detection)))
            valid = []
            for i, prev_particle in enumerate(self.particles):
                if prev_particle.is_init:
                    predicted = prev_particle.predict()
                for j, current_coord in enumerate(current_detection):
                    if prev_particle.is_init:
                        distance = np.sqrt((predicted[0] - current_coord["xcenter"]) ** 2 + (
                            predicted[1] - current_coord["ycenter"]) ** 2)
                    else:
                        distance = np.sqrt((prev_particle.attributes["xcenter"] -
                            current_coord["xcenter"]) ** 2 +
                            (prev_particle.attributes["ycenter"] - current_coord["ycenter"]) ** 2)

                    angle = np.abs(self.angle_difference(
                        prev_particle.attributes["orientation"], current_coord["orientation"]))
                    area = np.abs(prev_particle.attributes["area"] - current_coord["area"])
                    perim = np.abs(prev_particle.attributes["perim"] - current_coord["perim"])

                    if distance < self.params["maxDist"]:
                        cost[i, j] = self.compute_cost([distance, angle, area, perim], [
                                                    self.params["normDist"],
                                                    self.params["normAngle"],
                                                    self.params["normArea"],
                                                    self.params["normPerim"]])
                        valid.append((i, j))
                    else:
                        cost[i, j] = 1e34

            row, col = linear_sum_assignment(cost)

            assignment = []
            for i, _ in enumerate(self.particles):
                if i in row and (i, col[list(row).index(i)]) in valid:
                    assignment.append(col[list(row).index(i)])
                else:
                    assignment.append(-1)
        return assignment

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

    def update(self, current_detection: List[Dict[str, float]], order: List[int]) -> None:
        """Reassign current based on order.

        Parameters
        ---------
        order : list
            Reassingment

        Returns
        -------
        list
            Reordered current.

        """
        for i, part in enumerate(self.particles):
            if order[i] != -1:
                part.skip_count = 0
                part.attributes["time"] = self.im
                x = current_detection[order[i]]["xcenter"]
                y = current_detection[order[i]]["ycenter"]
                if part.is_init:
                    part.update(np.array([x, y]))
                else:
                    part.is_init = True
                    x_v = x - part.attributes["xcenter"]
                    y_v = y - part.attributes["ycenter"]
                    part.init_filter(np.array([x, y, x_v, y_v]))
                part.update_attributes(current_detection[order[i]], self.im)

        for i, curr in enumerate(current_detection):
            if i not in order:
                part = Particle(self.max_id)
                self.max_id += 1
                part.update_attributes(curr, self.im)
                self.particles.append(part)

    def clean(self, lost: List[int]) -> None:
        """Delete objects that were lost.
        Only counter is copied in this function. Other lists act as pointer.

        Parameters
        ----------
        lost : list
            Lost objects.
        """
        to_delete = []
        for i in lost:
            self.particles[i].skip_count += 1
            self.particles[i].update(None)
            if self.particles[i].skip_count > self.params["maxTime"]:
                to_delete.append(i)
        to_delete.sort(reverse=True)
        for i in to_delete:
            self.particles.pop(i)

    @staticmethod
    def hex_to_rgb(value):
        """Transform an hexadecimal image in RGB"""
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def make_verif_image(self, image: np.ndarray, tracking_data: List[Dict[str, float]]) -> None:
        """Make a image showing the detection."""
        colors = [ObjectTracker.hex_to_rgb(color) for color in mcolors.TABLEAU_COLORS.values()]
        image_to_draw = cv2.merge([image, image, image])
        font = cv2.FONT_HERSHEY_SIMPLEX
        for attributes in tracking_data:
            color = colors[int(attributes["id"]) % len(colors)]
            center = (int(attributes["xcenter"]), int(attributes["ycenter"]))
            center_writing = (int(attributes["xcenter"]),
                            int(attributes["ycenter"] + attributes["major_axis"]))
            image_to_draw = cv2.ellipse(img=image_to_draw,
                                center=center,
                                axes=(int(attributes["major_axis"]), int(attributes["minor_axis"])),
                                angle=180 * (1 - attributes["orientation"] / np.pi), startAngle=0, endAngle=360,
                                color=color, thickness=1)
            image_to_draw = cv2.putText(image_to_draw, str(attributes["id"]), org=center_writing, fontFace=font, fontScale=1, color=color)
        cv2.imwrite(os.path.join(self.save_folder, f"tracked{self.im:06d}.png"), image_to_draw)
