import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import pandas as pd

from pathlib import Path
from omegaconf.listconfig import ListConfig

METADATA_FILENAME = "metadata.csv"

# Debugging
N_FFT = 1024
N_MELS = 64


def save_signals(signals, sr, output_dir, log_melspectrogram=False):
    
    for i, signal in enumerate(signals):
        file_name = output_dir / f"{i}.wav"
        soundfile.write(file_name, signal, sr)

        if log_melspectrogram:
            file_name = output_dir / f"{i}.png"
            S = librosa.feature.melspectrogram(
                    signal, sr, n_fft=N_FFT, n_mels=N_MELS)
            librosa.display.specshow(S, x_axis='time',
                         y_axis='mel', sr=sr)

            plt.savefig(file_name)


def save_dataset_metadata(training_sample_configs, output_dir):
    
    def serialize_dict(d):
        serialized_dict = {}
        for key, value in d.items():
            if isinstance(value, int) or \
               isinstance(value, float) or \
               isinstance(value, str):
                serialized_dict[key] = value
            elif isinstance(value, Path) or \
                 isinstance(value, list) or \
                 isinstance(value, ListConfig):
                serialized_dict[key] = str(value)

        return serialized_dict

    output_dicts = [serialize_dict(d) for d in training_sample_configs]

    df = pd.DataFrame(output_dicts)
    df.to_csv(output_dir / METADATA_FILENAME)
