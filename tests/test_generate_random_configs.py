import numpy as np

from datasets.generate_random_configs import (
    generate_random_microphone_pair_coordinates
)


def test_generate_random_microphone_pair_coordinates():
    coords = generate_random_microphone_pair_coordinates([3,3,5], 0.5)

    coords = np.array(coords)

    assert np.sqrt(np.sum((coords[0] - coords[1])**2)) == 0.5
