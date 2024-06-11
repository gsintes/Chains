"""Module to detect chains in a track."""
from typing import List, Tuple, Any, Dict
from collections import defaultdict

import numpy as np
import pandas as pd

def get_ellipse_points(x: float, y: float, a: float, b: float, angle: float, num_points: int =100) -> np.ndarray:
    """
    Returns points on the ellipse perimeter for given parameters.
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    ellipse_points = np.array([a * np.cos(t), b * np.sin(t)])
    # Rotation matrix
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    rotated_points = R @ ellipse_points
    return np.array([x, y]).reshape(2, 1) + rotated_points

def project_ellipse_points(points: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    """
    Projects points of an ellipse onto a given axis.
    """
    projections = points.T @ axis
    return np.min(projections), np.max(projections)

def axes_to_check(ellipse1_points: np.ndarray, ellipse2_points: np.ndarray)-> np.ndarray:
    """
    Get the axes to check for the SAT algorithm.
    These are the normals to the edges of the convex hulls of the ellipses.
    """
    def edge_normals(points):
        edges = np.diff(points.T, axis=0)
        normals = np.array([-edges[:, 1], edges[:, 0]]).T
        return normals / np.linalg.norm(normals, axis=1).reshape(-1, 1)
    normals1 = edge_normals(ellipse1_points)
    normals2 = edge_normals(ellipse2_points)
    return np.vstack((normals1, normals2))

def ellipses_overlap(ellipse1: Tuple[float, float, float, float, float], ellipse2: Tuple[float, float, float, float, float]) -> bool:
    """
    Checks if two ellipses overlap using the Separating Axis Theorem.
    """
    x1, y1, a1, b1, t1 = ellipse1
    x2, y2, a2, b2, t2 = ellipse2
    ellipse1_points = get_ellipse_points(x1, y1, a1, b1, t1)
    ellipse2_points = get_ellipse_points(x2, y2, a2, b2, t2)
    for axis in axes_to_check(ellipse1_points, ellipse2_points):
        min1, max1 = project_ellipse_points(ellipse1_points, axis)
        min2, max2 = project_ellipse_points(ellipse2_points, axis)
        if max1 < min2 or max2 < min1:
            return False
    return True

def find_groups(data: pd.DataFrame)-> List[List[int]]:
    """
    Finds groups of connected ellipses.
    Parameters:
    :param data: DataFrame with columns id, xBody, yBody, bodyMajorAxisLength, bodyMinorAxisLength, tBody.
    :return: List of lists, each sub-list represents a group of connected ellipses.
    """
    ellipses = {row.id: (float(row.xBody), float(row.yBody), float(row.bodyMajorAxisLength), float(row.bodyMinorAxisLength), float(row.tBody))
                for row in data.itertuples(index=False)}
    # Create the graph
    ids = list(ellipses.keys())
    graph = defaultdict(list)
    for i, id1 in enumerate(ids):
        for id2 in ids[i+1:]:
            if ellipses_overlap(ellipses[id1], ellipses[id2]):
                graph[id1].append(id2)
                graph[id2].append(id1)
    visited = {id: False for id in ids}
    groups = []

    def dfs(node, visited, group):
        """Find connected components of a graph using depth-first search."""
        visited[node] = True
        group.append(node)
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor, visited, group)

    for id in ids:
        if not visited[id]:
            group: List[int] = []
            dfs(id, visited, group)
            groups.append(group)
    return groups

class ChainDetector:
    """Make the chains from the data."""
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.chain_data = pd.DataFrame(columns=["id", "chain_length", "imageNumber"])
        self.chain_dict: Dict[int, List[int]] = {}
        self.id: List[int] = []
        self.imageNumber: List[int] = []
        self.chain_length: List[int] = []
        self.bact_dict: Dict[int, int] = {}
        self.id_count = 0

    def prep_data(self, step: int) -> pd.DataFrame:
        """Prepare the data for the analysis."""
        data = self.data.copy()
        data = data[self.data.imageNumber == step]
        data["tBody"] = np.pi - data["tBody"]
        data["bodyMajorAxisLength"] = data["bodyMajorAxisLength"] + 4
        data["bodyMinorAxisLength"] = data["bodyMinorAxisLength"] * 0.75
        data = data[['id', 'xBody', 'yBody', 'bodyMajorAxisLength', 'bodyMinorAxisLength',
                               'tBody']]
        return data

    def initialize(self) -> None:
        """Initialize the analysis."""
        data = self.prep_data(0)
        groups = find_groups(data)
        for group in groups:
            self.chain_dict[self.id_count] = group
            for bact in group:
                self.bact_dict[bact] = self.id_count
            self.id_count += 1
        self.chain_length = [len(group) for group in groups]
        self.id = list(self.chain_dict.keys())
        self.imageNumber = [0 for _ in self.id]

    def clean_length(self) -> None:
        """Clean the chain length."""
        lengths = self.chain_data.groupby("id")["chain_length"].agg(pd.Series.mode)

    def process(self) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
        """Performs the analysis."""
        self.initialize()
        for step in range(1, self.data.imageNumber.max() + 1):
            data = self.prep_data(step)
            groups = find_groups(data)
            for group in groups:
                chain_ids = [self.bact_dict.get(bact, -1) for bact in group]
                set_chain_ids = list(set(chain_ids))
                if len(set_chain_ids) == 1:
                    if set_chain_ids[0] == -1:
                        id = self.id_count
                        for bact in group:
                            self.bact_dict[bact] = id
                        self.chain_dict[id] = group
                        self.id_count += 1
                    else:
                        id = set_chain_ids[0]
                    self.id.append(id)
                    self.chain_length.append(len(group))
                    self.imageNumber.append(step)

                else:
                    print(step, chain_ids, group)
                    raise Exception("Error in the chain detection.")
        self.chain_data = pd.DataFrame({"id": self.id, "chain_length": self.chain_length, "imageNumber": self.imageNumber})
        self.clean_length()
        return self.chain_data, self.chain_dict
