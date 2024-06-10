"""Module to detect chains in a track."""
from typing import List, Tuple, Callable, Any
from collections import defaultdict

import numpy as np
import pandas as pd

from tracking_analysis import Analysis


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

def ellipses_overlap(ellipse1: Tuple[float, float, float, float], ellipse2: Tuple[float, float, float, float]) -> bool:
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



def find_groups(data: pd.DataFrame, are_connected: Callable[[Tuple, Tuple], bool])-> List[List[int]]:
    """
    Finds groups of connected ellipses.
    Parameters:
    :param data: DataFrame with columns id, xBody, yBody, bodyMajorAxisLength, bodyMinorAxisLength, tBody.
    :param are_connected: Function that takes two ellipses and returns True if they overlap, otherwise False.
    :return: List of lists, each sub-list represents a group of connected ellipses.
    """
    ellipses = {row.id: (row.xBody, row.yBody, row.bodyMajorAxisLength, row.bodyMinorAxisLength, row.tBody)
                for row in data.itertuples(index=False)}
    # Create the graph
    ids = list(ellipses.keys())
    graph = defaultdict(list)
    for i, id1 in enumerate(ids):
        for id2 in ids[i+1:]:
            if are_connected(ellipses[id1], ellipses[id2]):
                graph[id1].append(id2)
                graph[id2].append(id1)
    visited = [False] * len(ids)
    groups = []
    def dfs(node, visited, group):
        """Find connected components of a graph using depth-first search."""
        visited[node] = True
        group.append(node)
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor, visited, group)

    for i in range(len(ids)):
        if not visited[i]:
            group: List[int] = []
            dfs(i, visited, group)
            groups.append(group)
    return groups



if __name__ == "__main__":
    ana = Analysis("/Users/sintes/Desktop/ImageSeq")
    ana.calculate_velocity()
    data = ana.clean(data=ana.data_ind, scale=6.24, frame_rate=30, bactLength=10)
    data["bodyMajorAxisLength"] = data["bodyMajorAxisLength"] + 4
    data["tBody"] = np.pi - data["tBody"]
    data = data[data.imageNumber == 0]
    data = data[['id', 'xBody', 'yBody', 'bodyMajorAxisLength', 'bodyMinorAxisLength',
       'tBody']]

    # print(data)
    print(find_groups(data, ellipses_overlap))