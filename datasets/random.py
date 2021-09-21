import librosa
import numpy as np
import random

from omegaconf.listconfig import ListConfig

from datasets.math_utils import compute_distance, compute_tdoa, normalize_tdoa
DEFAULT_DEVICE_HEIGHT = 1


def generate_sample_config(base_config):
    room_dims = base_config["room_dims"]
    mic_coordinates = base_config["mic_coordinates"]
    source_coordinates = base_config["source_coordinates"]
    mic_distance = base_config["mic_distance"]
    sr = base_config["base_sampling_rate"]

    if not mic_coordinates:
        mic_coordinates = generate_random_microphone_coordinates(
                                                    room_dims, mic_distance)
    if not source_coordinates:
        source_coordinates = generate_random_points(room_dims, 1)[0]
    mic_delays, mic_sampling_rates, mic_gains = \
        generate_random_microphone_signal_distortions(base_config)

    # Make signal's duration bigger to trim silent beginning
    max_delay = max(mic_delays)
    total_duration = base_config["n_microphone_seconds"] + max_delay
    if "speech_samples" in base_config:
        source_signal, source_gain = generate_random_speech_signal(
                                        total_duration, sr, base_config["speech_samples"])
    else:
        source_signal, source_gain = generate_random_signal(int(sr*total_duration))

    tdoa, normalized_tdoa = _compute_tdoa(source_coordinates, mic_coordinates)

    return {
        "room_dims": room_dims,
        "source_coordinates": source_coordinates,
        "mic_coordinates": mic_coordinates,
        "mic_delays": mic_delays,
        "mic_gains": mic_gains,
        "mic_sampling_rates": mic_sampling_rates,
        "n_microphone_seconds": base_config["n_microphone_seconds"],
        "tdoa": tdoa,
        "normalized_tdoa": normalized_tdoa,
        "sr": sr,
        "source_signal": source_signal,
        "source_gain": source_gain,
        "trim_beginning": base_config["trim_beginning"],
        "anechoic": base_config["anechoic"]
    }


def generate_random_points(room_dims,
                           n_points,
                           height=DEFAULT_DEVICE_HEIGHT):
    return [
        [
            random.uniform(0, room_dims[0]),
            random.uniform(0, room_dims[1]),
            height
        ]
        for _ in range(n_points)
    ]


def generate_random_microphone_coordinates(room_dims,
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


def generate_random_signal(n_samples:int, random_gain=True):
    """Generate a random signal to be emmited by the source.
    The signal is white gaussian noise distributed.
    The signal is also multiplied by an uniformly distributed gain to simulate
    the unknown source gain.
    """
    
    gain = np.random.uniform() if random_gain else 1
    source_signal = np.random.normal(size=n_samples)*gain

    return source_signal, gain


def generate_random_microphone_signal_distortions(base_config):
    
    mic_delays = generate_random_delay_in_ms(base_config["n_mics"],
                                             base_config["mic_delay_ranges"])
    
    mic_sampling_rates = generate_random_sampling_rate(base_config["n_mics"],
                                               base_config["mic_sampling_rate_ranges"])

    mic_gains = generate_random_gain(base_config["n_mics"],
                                     base_config["mic_gain_ranges"])


    return mic_delays, mic_sampling_rates, mic_gains


def generate_random_speech_signal(signal_duration, sr, speech_file_paths):
    total_duration_in_samples = int(signal_duration*sr)
    random_file_path = random.choice(speech_file_paths)
    source_signal, _ = librosa.load(random_file_path, sr=sr)
    # Improvement: Selecting the starting sample randomly instead of using 0
    source_signal = source_signal[:total_duration_in_samples]
    # For gaussian noise, a random gain is applied. For speech this isn't done.
    source_gain = 1

    return source_signal, source_gain


def generate_random_delay_in_ms(n, delay_interval):
    "Generate random delay in milliseconds"
    if type(delay_interval) == list or type(delay_interval) == ListConfig:
        return [
            random.uniform(delay_interval[0], delay_interval[1])/1000
            for _ in range(n)
        ]
    elif type(delay_interval) == int:
        return [delay_interval/1000]*n


def generate_random_sampling_rate(n, sampling_rate_interval):
    if type(sampling_rate_interval) == list or type(sampling_rate_interval) == ListConfig:
        return [
            random.randint(sampling_rate_interval[0], sampling_rate_interval[1])
            for _ in range(n)
        ]
    elif type(sampling_rate_interval) == int:
        return [sampling_rate_interval]*n


def generate_random_gain(n, gain):
    "Generate random delay in milliseconds"
    if type(gain) == list or type(gain) == ListConfig:
        if type(gain[0]) == list or type(gain[0]) == ListConfig:
            return [
                random.uniform(gain[i][0], gain[i][1])
                for i in range(n)
            ]
        return [
            random.uniform(gain[0], gain[1])
            for _ in range(n)
        ]
    elif type(gain) == int:
        return [gain]*n


def _is_point_in_room(room_dims, point):
    for i, dim in enumerate(room_dims):
        if point[i] < 0 or point[i] >= dim:
            return False
    return True


def _compute_tdoa(source_coordinates, mic_coordinates):
    mic_distance = compute_distance(mic_coordinates[0], mic_coordinates[1])
    tdoa = compute_tdoa(source_coordinates, mic_coordinates)
    normalized_tdoa = normalize_tdoa(tdoa, mic_distance)

    return tdoa, normalized_tdoa
