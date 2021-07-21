from datasets.settings import SPEED_OF_SOUND
import numpy as np


def compute_distance(p1, p2):
    "Compute the euclidean distance between two points"

    p1 = np.array(p1)
    p2 = np.array(p2)

    return np.linalg.norm(p1 - p2)


def compute_tdoa(source, microphones):
    dist_0 = compute_distance(source, microphones[0])
    dist_1 = compute_distance(source, microphones[1])

    return (dist_0 - dist_1)/SPEED_OF_SOUND


def compute_tdoa_range(mic_coordinates):
    # The maximum possible TDOA is when the source is on the line that
    # crosses both microphones
    max_tdoa = compute_distance(mic_coordinates[0],
                                mic_coordinates[1])/SPEED_OF_SOUND
    min_tdoa = -max_tdoa

    return min_tdoa, max_tdoa


def normalize(x, min_x, max_x):
    return (x - min_x)/(max_x - min_x)
