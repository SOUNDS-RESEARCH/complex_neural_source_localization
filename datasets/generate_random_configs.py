import numpy as np
import random

DEFAULT_DEVICE_HEIGHT = 1


def generate_random_source_coordinates(room_dims, height=DEFAULT_DEVICE_HEIGHT):
    source_x = random.uniform(0, room_dims[0])
    source_y = random.uniform(0, room_dims[1])
    source_coordinates = [source_x, source_y, height]

    return source_coordinates


def generate_random_microphone_coordinates(room_dims,
                                           n_mics,
                                           height=DEFAULT_DEVICE_HEIGHT):
    return [
        [
            random.uniform(0, room_dims[0]),
            random.uniform(0, room_dims[1]),
            height
        ]
        for i in range(n_mics)
    ]


def generate_random_source_signal(sr: int, sample_duration_in_secs: float,
                                  random_gain=True):
    """Generate a random signal to be emmited by the source.
    The signal is white gaussian noise distributed.
    The signal is also multiplied by an uniformly distributed gain to simulate
    the unknown source gain.
    """
    num_samples = int(sr*sample_duration_in_secs)
    gain = np.random.uniform() if random_gain else 1
    source_signal = np.random.normal(size=num_samples)*gain

    return source_signal, gain


def generate_random_sampling_rate(low, high):
    return random.randint(low, high)


def generate_random_delay(low, high):
    "Generate random delay in milliseconds"
    return random.uniform(low, high)/1000


def generate_random_gain(low, high):
    return random.uniform(low, high)
