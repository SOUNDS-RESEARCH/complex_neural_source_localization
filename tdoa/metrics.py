import numpy as np

from datasets.math_utils import compute_distance


def compute_error(candidate, mic_1, mic_2, tdoa, norm="l1"):
    "Get a measure of how far a candidate point (x, y) is from a computed doa"

    dist_1 = compute_distance(candidate, mic_1)
    dist_2 = compute_distance(candidate, mic_2)

    error = tdoa - np.abs(dist_1 - dist_2)

    if norm == "l2":
        error = error**2
    elif norm == "l1":
        error = np.abs(error)
    
    return error
