import numpy as np
import random

DEFAULT_DEVICE_HEIGHT = 1


def generate_random_source_coordinates(room_dims, height=DEFAULT_DEVICE_HEIGHT):
    source_x = random.uniform(0, room_dims[0])
    source_y = random.uniform(0, room_dims[1])
    source_coordinates = [source_x, source_y, height]

    return source_coordinates


def generate_random_microphone_coordinates(room_dims,
                                            height=DEFAULT_DEVICE_HEIGHT):
    mic_1_x = random.uniform(0, room_dims[0])
    mic_2_x = random.uniform(0, room_dims[0])

    mic_1_y = random.uniform(0, room_dims[1])
    mic_2_y = random.uniform(0, room_dims[1])

    return [
        [mic_1_x, mic_1_y, height],
        [mic_2_x, mic_2_y, height],
    ]


def generate_random_source_signal(base_config):
    sampling_rate = base_config["base_sampling_rate"]
    mic_delays = base_config["mic_delays"]
    sample_duration_in_secs = base_config["sample_duration_in_secs"]

    max_delay = max(mic_delays)
    total_duration = sample_duration_in_secs + max_delay
    num_samples = sampling_rate*int(total_duration)
    gain = np.random.uniform()
    source_signal = np.random.normal(size=num_samples)*gain

    return source_signal, gain
