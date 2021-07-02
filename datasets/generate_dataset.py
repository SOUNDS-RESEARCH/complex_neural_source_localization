import librosa.display
import numpy as np
import os
import pandas as pd
import random
import pyroomacoustics as pra
import soundfile

import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from datasets.math_utils import normalize, compute_distance
from datasets.settings import (
    SAMPLE_DURATION_IN_SECS,
    SPEED_OF_SOUND,
    SR,
    ROOM_DIMS,
    MIC_POSITIONS,
    N_SAMPLES, 
    DEFAULT_OUTPUT_DATASET_DIR
)
from neural_tdoa.models.settings import (
    N_FFT, N_MELS, HOP_LENGTH
)

METADATA_FILENAME = "metadata.csv"
DEFAULT_DEVICE_HEIGHT = 1


def generate_dataset(
    output_dir=DEFAULT_OUTPUT_DATASET_DIR,
    room_dims=ROOM_DIMS,
    mic_positions=MIC_POSITIONS,
    num_samples=N_SAMPLES,
    sample_duration_in_secs=SAMPLE_DURATION_IN_SECS,
    sr=SR,
    save_melspectogram=False):

    output_dir = Path(output_dir)
    output_samples_dir = output_dir / "samples"

    os.makedirs(output_samples_dir, exist_ok=True)

    experiment_configs = []
    for num_sample in tqdm(range(num_samples)):
        experiment_config = _generate_random_experiment_settings(
            room_dims, mic_positions, sr, sample_duration_in_secs)
        
        experiment_config["signals_dir"] = output_samples_dir / str(num_sample)
        experiment_configs.append(experiment_config)

        _generate_sample(experiment_config, save_melspectogram)

    _save_experiment_metadata(experiment_configs, output_dir)


def _simulate(experiment_settings):

    room = pra.ShoeBox(experiment_settings["room_dims"],
                       fs=experiment_settings["sr"])

    room.add_microphone_array(experiment_settings["mic_coordinates"])

    room.add_source(experiment_settings["source_coordinates"],
        experiment_settings["source_signal"]
    )

    room.simulate()

    signals = room.mic_array.signals
    
    return signals

def _generate_random_experiment_settings(room_dims=ROOM_DIMS,
                                         microphone_coordinates=None,
                                         sr=SR,
                                         sample_duration_in_secs=SAMPLE_DURATION_IN_SECS):

    if microphone_coordinates is None:
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

    num_samples = sr*sample_duration_in_secs
    gain = np.random.uniform()
    source_signal = np.random.normal(size=num_samples)*gain
    return {
        "room_dims": room_dims,
        "source_x": source_x,
        "source_y": source_y,
        "source_coordinates": np.array(source_coordinates).T,
        "mic_coordinates": np.array(microphone_coordinates).T,
        "tdoa": tdoa,
        "normalized_tdoa": normalize(tdoa, min_tdoa, max_tdoa),
        "num_samples": num_samples,
        "sr": sr,
        "source_signal": source_signal,
        "gain": gain
    }


def _generate_sample(experiment_settings, save_melspectogram=False):
    output_signals = _simulate(experiment_settings)

    os.makedirs(experiment_settings["signals_dir"], exist_ok=True)
    _save_signals(output_signals,
                  experiment_settings["sr"],
                  experiment_settings["signals_dir"],
                  save_melspectogram)


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


def _save_signals(signals, sr, output_dir, save_melspectogram=False):
    
    for i, signal in enumerate(signals):
        file_name = output_dir / f"{i}.wav"
        soundfile.write(file_name, signal, sr)

        if save_melspectogram:
            file_name = output_dir / f"{i}.png"
            S = librosa.feature.melspectrogram(
                    signal, SR, n_fft=N_FFT, n_mels=N_MELS)
            librosa.display.specshow(S, x_axis='time',
                         y_axis='mel', sr=SR)

            plt.savefig(file_name)

def _save_experiment_metadata(experiment_configs, output_dir):
    output_keys = [
        "source_x", "source_y", "tdoa", "normalized_tdoa", "signals_dir"
    ]
    
    def filter_keys(experiment_config):
        output_dict = {
            key: value for key, value in experiment_config.items()
            if key in output_keys
        }

        return output_dict
    
    output_dicts = [filter_keys(exp) for exp in experiment_configs]

    df = pd.DataFrame(output_dicts)
    df.to_csv(output_dir / METADATA_FILENAME)


if __name__ == "__main__":
    generate_dataset()
