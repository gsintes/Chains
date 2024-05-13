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
        
        self.object_id = id 
        self.attributes: Dict[str, float] = {}

        self.KF = KalmanFilter(4, 2)
        self.KF.F = np.array([[1., 0., 1., 0.],
                              [0., 1., 0., 1.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]])
        self.KF.H = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.]])
        self.KF.P *= 1000. #TODO fix value for the 3 next param
        self.KF.R = [[2, 2]] # Error in estimate
        self.KF.Q = Q_discrete_white_noise(dim=4, dt=0.1, var=0.13)

        self.skip_count = 0 
        self.line = [] 

    def update_attributes(self, detection: Dict[str, Union[float, int]]) -> None:
        """Update the attributes based on the detection."""
        self.attributes = detection
        
class ObjectTracker(object):
    """Track the objects in a video and solve the assignment issue."""
    def __init__(self, params: Dict[str, float]= {},save_folder: str = "", visualization: bool = True):
        self.objects: List[Particle] = []
        self.object_id = 0
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
            detection = self.detector.process(image1)
            for detec in detection:
                
                particle = Particle(self.object_id)
                particle.update_attributes(detec)
                self.objects.append(particle)
                
                self.object_id += 1
            return self.prev_detection
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
    
    def update(self, detections):
        """Update the tracking."""

        N , M = len(self.objects), len(detections)
        
        cost_matrix = np.zeros(shape=(N, M)) 
        
        for i in range(N):
            for j in range(M):
                diff_distance = self.objects[i].prediction - detections[j]
                dist = diff_distance[0][0]*diff_distance[0][0] +diff_distance[1][0]*diff_distance[1][0]
                cost_matrix[i][j] = self.compute_cost()

        assign = []
        for _ in range(N):
            assign.append(-1)
            
        rows, cols = linear_sum_assignment(cost_matrix)
        
        for i in range(len(rows)):
            assign[rows[i]] = cols[i]

        unassign = []
        for i in range(len(assign)):
            if (assign[i] != -1):
                if (cost_matrix[i][assign[i]] > self.params["min_dist"]):
                    assign[i] = -1
                    unassign.append(i)
            else:
                self.objects[i].skip_count += 1

        del_objects = []
        for i in range(len(self.objects)):
            if (self.objects[i].skip_count > self.params["max_skip"]):
                del_objects.append(i)

        if del_objects: # TODO check
            for id in del_objects:
                if id < len(self.objects):
                    del self.objects[id]
                    del assign[id]         

        for i in range(len(detections)):
                if i not in assign:
                    self.objects.append(Particle(detections[i], self.object_id))
                    self.object_id += 1


                
        for i in range(len(assign)):
            self.objects[i].KF.predict()

            if(assign[i] != -1):
                self.objects[i].skip_count = 0
                self.objects[i].prediction = self.objects[i].KF.update(detections[assign[i]])
            else:
                self.objects[i].prediction = self.objects[i].KF.update( np.array([[0], [0]]))