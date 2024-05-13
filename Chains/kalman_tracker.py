"""Tracker based on Kalman filters."""

from typing import List, Dict, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from nptyping import NDArray, Float, Shape
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from base_detector import BaseDetector


class Particle:
    """A particle is an object tracked in the image."""
    def __init__(self, id: int):
        self.is_init = False
        self.id = id 
        self.attributes: Dict[str, Union[float, int]] = {}

        self._KF = KalmanFilter(4, 2)
        self._KF.F = np.array([[1., 0., 1., 0.],
                              [0., 1., 0., 1.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]])
        self._KF.H = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.]])
        self._KF.P *= 1000. #TODO fix value for the 3 next param
        self._KF.R = [[2, 2]] # Error in estimate
        self._KF.Q = Q_discrete_white_noise(dim=4, dt=0.1, var=0.13)

        self.skip_count = 0 
        self.line = [] 

    def init_filter(self, state: NDArray[Shape[4, Float]]) -> None:
        """Initialise the filter."""
        self.is_init = True
        self._KF.x = state

    def predict(self) -> NDArray[Shape[4, Float]]:
        """Predict the position."""
        self._KF.predict()
        return self._KF.x
    
    def update(self, measure: NDArray[Shape[2, Float]]) -> None:
        """Predict the position."""
        self._KF.update(measure)

    def update_attributes(self, detection: Dict[str, Union[float, int]]) -> None:
        """Update the attributes based on the detection."""
        self.attributes = detection
        
class ObjectTracker(object):
    """Track the objects in a video and solve the assignment issue."""
    def __init__(self, params: Dict[str, float]= {},save_folder: str = "", visualization: bool = True):
        self.particles: List[Particle] = []
        self.max_id = 0
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

    def initialize(self, image1: np.ndarray, image2: np.ndarray) -> None:
        """Initialize the tracker on two first images.

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
            self.id: List[int] = []
            self.is_init = True
            self.prev_detection = self.detector.process(image1)
            self.lost: List[int] = []
            self.im = 0
            for detec in self.prev_detection:
                particle = Particle(self.max_id)
                particle.update_attributes(detec)
                particle.attributes["time"] = self.im
                particle.attributes["id"] = particle.id
                self.particles.append(particle)
                
                self.max_id += 1

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
    
    def process(self, image: np.ndarray) -> List[Dict[str, Union[int, float]]]:
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
                j["time"] = self.im
                j["id"] = self.id[i]
            self.im += 1
            self.prev_detection = self.current_detection
            self.lost_ids = losts
            if self.visualization:
                self.make_verif_image(image)
            self.count += 1
            
            return [j for i, j in enumerate(self.current_detection) if i not in losts]
        return []

    def assign(self) -> List[int]:
        """Find the optimal assignent.
        Returns
        -------
        list
            Assignment.
        """
        if len(self.particles) == 0:
            assignment: List[int] = []
        elif len(self.current_detection) == 0:
            assignment = [-1] * len(self.particles)
        else:
            cost = np.zeros((len(self.particles), len(self.current_detection)))
            valid = []
            for i, prev_particle in enumerate(self.particles):
                for j, current_coord in enumerate(self.current_detection):
                    if prev_particle.is_init:
                        predicted = prev_particle.predict()
                        distance = np.sqrt((predicted[0] - current_coord["xcenter"]) ** 2 + (
                            predicted[1] - current_coord["ycenter"]) ** 2)
                    else:
                        distance = np.sqrt((prev_particle.attributes["xcenter"] - current_coord["xcenter"]) ** 2 + (
                            prev_particle.attributes["xcenter"] - current_coord["ycenter"]) ** 2)
                    
                    angle = np.abs(self.angle_difference(
                        prev_particle.attributes["orientation"], current_coord["orientation"]))
                    area = np.abs(prev_particle.attributes["area"] - current_coord["area"])
                    perim = np.abs(prev_particle.attributes["perim"] - current_coord["perim"])

                    if distance < self.params["maxDist"]:
                        cost[i, j] = self.compute_cost([distance, angle, area, perim], [
                                                    self.params["normDist"], self.params["normAngle"], self.params["normArea"], self.params["normPerim"]])
                        valid.append((i, j))
                    else:
                        cost[i, j] = 1e34

            row, col = linear_sum_assignment(cost)

            assignment = []
            for i, _ in enumerate(self.prev_detection):
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
    
    def reassign(self,
                 past: List[Dict[str, Union[int, float]]],
                 current: List[Dict[str, Union[int, float]]],
                 order: List[int]) -> List[Dict[str, Union[int, float]]]:
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

    def update(self, detections):
        """Update the tracking."""

        del_objects = []
        for i in range(len(self.particles)):
            if (self.particles[i].skip_count > self.params["max_skip"]):
                del_objects.append(i)

        if del_objects: # TODO check
            for id in del_objects:
                if id < len(self.particles):
                    del self.particles[id]
                    del assign[id]         

        for i in range(len(detections)):
                if i not in assign:
                    self.particles.append(Particle(detections[i], self.object_id))
                    self.object_id += 1


                
        for i in range(len(assign)):
            self.particles[i]._KF.predict()

            if(assign[i] != -1):
                self.particles[i].skip_count = 0
                self.particles[i].prediction = self.particles[i]._KF.update(detections[assign[i]])
            else:
                self.particles[i].prediction = self.particles[i]._KF.update( np.array([[0], [0]]))