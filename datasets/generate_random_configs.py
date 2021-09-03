import librosa
import numpy as np
import random

DEFAULT_DEVICE_HEIGHT = 1


def generate_random_points(room_dims,
                           n_points,
                           height=DEFAULT_DEVICE_HEIGHT):
    return [
        [
            random.uniform(0, room_dims[0]),
            random.uniform(0, room_dims[1]),
            height
        ]
        for i in range(n_points)
    ]


def generate_random_microphone_pair_coordinates(room_dims,
                                                distance=None,
                                                height=DEFAULT_DEVICE_HEIGHT):
    """Generate random coordinates (x, y, height) for a pair of microphones inside a room.
    Optionally the separation between the microphones may be fixed by providing the 'distance'
    parameter. Note that the height is not randomized.

    To generate microphone positions with a fixed distance, the following procedure was used:

    1. Generate microphone 1's coordinates randomly within the room
    2. Generate a random azimuth between [0, 2*pi]
    3. Place a point at angle azimuth within a circle with radius=distance
    4. Generate microphone 2's coordinates by summing results obtained in 1 and 3
    5. If all points are within the room, finish. Else, go to 1  


    Args:
        room_dims (list): Width, length and height of the room.
        distance (float, optional): If provided, microphones will always be separated by this distance
        height (float, optional): [description]. Defaults to DEFAULT_DEVICE_HEIGHT.
    """

    if distance is None:
        # Simply generate random positions
        return generate_random_points(room_dims, 2, height=height)

    while True:    
        m1 = generate_random_points(room_dims, 1, height)[0]
        azimuth = random.uniform(0, 2*np.pi)
        distance_vector = [distance*np.cos(azimuth), distance*np.sin(azimuth)]
        m2 = [m1[0] + distance_vector[0], m1[1] + distance_vector[1], height]

        if not _is_point_in_room(room_dims, m2):
            continue
        return [m1, m2]


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


def generate_random_speech_signal(signal_duration, sr, speech_file_paths):
    total_duration_in_samples = int(signal_duration*sr)
    random_file_path = random.choice(speech_file_paths)
    source_signal, _ = librosa.load(random_file_path, sr=sr)
    # Improvement: Selecting the starting sample randomly instead of using 0
    source_signal = source_signal[:total_duration_in_samples]
    # For gaussian noise, a random gain is applied. For speech this isn't done.
    source_gain = 1

    return source_signal, source_gain


def generate_random_sampling_rate(low, high):
    return random.randint(low, high)


def generate_random_delay(low, high):
    "Generate random delay in milliseconds"
    return random.uniform(low, high)/1000


def generate_random_gain(low, high):
    return random.uniform(low, high)


def _is_point_in_room(room_dims, point):
    for i, dim in enumerate(room_dims):
        if point[i] < 0 or point[i] >= dim:
            return False
    return True