import numpy as np

from datasets.random import (
    generate_random_microphone_coordinates
)


def test_generate_random_microphone_coordinates():
    coords = generate_random_microphone_coordinates([3,3,5], 0.5)

    coords = np.array(coords)

    assert np.sqrt(np.sum((coords[0] - coords[1])**2)) == 0.5
