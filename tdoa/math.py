import numpy as np


def compute_distance(p1, p2):
    "Compute the euclidean distance between two points"

    p1 = np.array(p1)
    p2 = np.array(p2)

    return np.linalg.norm(p1 - p2)
