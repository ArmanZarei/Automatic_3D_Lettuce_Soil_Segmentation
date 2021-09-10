import numpy as np


def normalize_points(points):
    points = points - points.mean(axis=0)
    points /= np.linalg.norm(points, axis=1).max()

    return points