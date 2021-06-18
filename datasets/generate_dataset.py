import os
import numpy as np
import pandas as pd
import random
import soundfile

from pathlib import Path
from tqdm import tqdm

import pyroomacoustics as pra
from datasets.settings import (
    SAMPLE_DURATION_IN_SECS,
    SPEED_OF_SOUND,
    SR,
    ROOM_DIMS,
    MIC_POSITIONS,
    N_SAMPLES, 
    SOURCE_HEIGHT,
    DEFAULT_OUTPUT_DATASET_DIR
)


def generate_dataset(
    output_dir=DEFAULT_OUTPUT_DATASET_DIR,
    room_dims=ROOM_DIMS,
    mic_positions=MIC_POSITIONS,
    num_samples=N_SAMPLES,
    source_height=SOURCE_HEIGHT,
    sample_duration_in_secs=SAMPLE_DURATION_IN_SECS,
    sr=SR):

    output_dir = Path(output_dir)
    output_samples_dir = output_dir / "samples"

    os.makedirs(output_samples_dir, exist_ok=True)

    max_tdoa = _compute_distance(mic_positions[0], mic_positions[1])
    min_tdoa = -max_tdoa

    samples_data = []
    for num_sample in tqdm(range(num_samples)):
        source_x = random.uniform(0, room_dims[0])
        source_y = random.uniform(0, room_dims[1])
        source_position = [source_x, source_y, source_height]

        output_signals = _simulate(room_dims,
                                   source_position,
                                   mic_positions,
                                   sr,
                                   sample_duration_in_secs)

        output_sample_dir = output_samples_dir / str(num_sample)
        os.makedirs(output_sample_dir, exist_ok=True)

        for i, signal in enumerate(output_signals):
            file_name = output_sample_dir / f"{i}.wav"
            soundfile.write(file_name, signal, sr)

        tdoa = _compute_tdoa(source_position, mic_positions)
        samples_data.append({
            "signals_dir": output_sample_dir,
            "source_x": source_x,
            "source_y": source_y,
            "tdoa": tdoa,
            "normalized_tdoa": _normalize(tdoa, min_tdoa, max_tdoa)
        })
    
    df = pd.DataFrame(samples_data)
    
    df.to_csv(output_dir / "metadata.csv")

def _simulate(room_dims,
              source_position,
              microphone_positions,
              sr,
              sample_duration_in_secs):

    room = pra.ShoeBox(room_dims,
                       fs=sr)

    room.add_microphone_array(np.array(microphone_positions).T)

    num_samples = sr*sample_duration_in_secs
    source_signal = np.random.normal(size=num_samples)
    room.add_source(np.array(source_position).T, source_signal)

    room.simulate()

    signals = room.mic_array.signals
    
    return signals


def _compute_tdoa(source, microphones):
    dist_0 = _compute_distance(source, microphones[0])
    dist_1 = _compute_distance(source, microphones[1])

    return (dist_0 - dist_1)/SPEED_OF_SOUND


def _compute_distance(p1, p2):
    "Compute the euclidean distance between two points"

    p1 = np.array(p1)
    p2 = np.array(p2)

    return np.linalg.norm(p1 - p2)


def _normalize(x, min_x, max_x):
    return (x - min_x)/(max_x - min_x)


if __name__ == "__main__":
    generate_dataset()
