from math import exp
import numpy as np
import os
import random
from pathlib import Path
from tqdm import tqdm

from pyroomasync import ConnectedShoeBox, simulate

from datasets.logger import save_experiment_metadata, save_signals
from datasets.math_utils import normalize, compute_distance
from datasets.settings import (
    SAMPLE_DURATION_IN_SECS,
    SPEED_OF_SOUND,
    SR,
    ROOM_DIMS,
    MIC_POSITIONS,
    MIC_SAMPLING_RATES,
    MIC_DELAYS,
    N_SAMPLES, 
    DEFAULT_OUTPUT_DATASET_DIR
)

DEFAULT_DEVICE_HEIGHT = 1


def generate_dataset(output_dir=DEFAULT_OUTPUT_DATASET_DIR,
                     room_dims=ROOM_DIMS,
                     mic_positions=MIC_POSITIONS,
                     num_samples=N_SAMPLES,
                     sample_duration_in_secs=SAMPLE_DURATION_IN_SECS,
                     mic_sampling_rates=MIC_SAMPLING_RATES,
                     base_sampling_rate=SR,
                     mic_delays=MIC_DELAYS,
                     log_melspectrogram=False):

    output_dir = Path(output_dir)
    output_samples_dir = output_dir / "samples"
    os.makedirs(output_samples_dir, exist_ok=True)

    base_experiment_config = {
        "room_dims": room_dims,
        "mic_coordinates": mic_positions,
        "mic_sampling_rates": mic_sampling_rates,
        "mic_delays":mic_delays,
        "sample_duration_in_secs": sample_duration_in_secs,
        "base_sampling_rate": base_sampling_rate
    }

    experiment_configs = []
    for num_sample in tqdm(range(num_samples)):
        experiment_config = _generate_random_experiment_settings(
                                base_experiment_config)
        
        experiment_config["signals_dir"] = output_samples_dir / str(num_sample)
        experiment_configs.append(experiment_config)

        _generate_sample(experiment_config,
                         log_melspectrogram=log_melspectrogram)

    save_experiment_metadata(experiment_configs, output_dir)


def _simulate(experiment_settings):
    base_sr = experiment_settings["sr"]

    room = ConnectedShoeBox(experiment_settings["room_dims"], fs=base_sr)

    room.add_microphone_array(experiment_settings["mic_coordinates"],
                              delay=experiment_settings["mic_delays"])

    room.add_source(experiment_settings["source_coordinates"],
        experiment_settings["source_signal"]
    )

    signals = simulate(room)

    max_delay = max(experiment_settings["mic_delays"])
    max_delay_in_samples = int(max_delay*base_sr)

    # Remove silence in the beginning of signals which might make
    # Detecting the delays "too easy", then truncate to experiment output size
    signals = signals[:, max_delay_in_samples:]
    signals = signals[:, :experiment_settings["num_samples"]]
    
    return signals

def _generate_random_experiment_settings(base_config):
    microphone_coordinates = base_config["mic_coordinates"]
    room_dims = base_config["room_dims"]
    sample_duration_in_secs = base_config["sample_duration_in_secs"]
    base_sampling_rate = base_config["base_sampling_rate"]
    mic_delays = base_config["mic_delays"]
    max_delay = max(mic_delays)
    total_duration = sample_duration_in_secs + max_delay

    if base_config["mic_coordinates"] is None:
        microphone_coordinates = _generate_random_microphone_coordinates(room_dims)

    max_tdoa = compute_distance(
                    microphone_coordinates[0],
                    microphone_coordinates[1]
    )/SPEED_OF_SOUND
    min_tdoa = -max_tdoa

    source_x = random.uniform(0, room_dims[0])
    source_y = random.uniform(0, room_dims[1])
    source_coordinates = [source_x, source_y, microphone_coordinates[0][2]]

    tdoa = _compute_tdoa(source_coordinates, microphone_coordinates)

    num_samples = base_sampling_rate*int(total_duration)
    gain = np.random.uniform()
    source_signal = np.random.normal(size=num_samples)*gain
    return {
        "room_dims": room_dims,
        "source_x": source_x,
        "source_y": source_y,
        "source_coordinates": source_coordinates,
        "mic_coordinates": microphone_coordinates,
        "mic_delays": mic_delays,
        "tdoa": tdoa,
        "normalized_tdoa": normalize(tdoa, min_tdoa, max_tdoa),
        "num_samples": num_samples,
        "sr": base_sampling_rate,
        "source_signal": source_signal,
        "gain": gain
    }


def _generate_sample(experiment_settings, log_melspectrogram=False):
    output_signals = _simulate(experiment_settings)

    os.makedirs(experiment_settings["signals_dir"], exist_ok=True)
    save_signals(output_signals,
                  experiment_settings["sr"],
                  experiment_settings["signals_dir"],
                  log_melspectrogram)


def _generate_random_microphone_coordinates(room_dims,
                                            height=DEFAULT_DEVICE_HEIGHT):
    mic_1_x = random.uniform(0, room_dims[0])
    mic_2_x = random.uniform(0, room_dims[0])

    mic_1_y = random.uniform(0, room_dims[1])
    mic_2_y = random.uniform(0, room_dims[1])

    return [
        [mic_1_x, mic_1_y, height],
        [mic_2_x, mic_2_y, height],
    ]


def _compute_tdoa(source, microphones):
    dist_0 = compute_distance(source, microphones[0])
    dist_1 = compute_distance(source, microphones[1])

    return (dist_0 - dist_1)/SPEED_OF_SOUND


if __name__ == "__main__":
    generate_dataset()
