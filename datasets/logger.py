import librosa.display
import matplotlib.pyplot as plt
import soundfile
import pandas as pd

from datasets.settings import SR
from neural_tdoa.settings import N_FFT, N_MELS


METADATA_FILENAME = "metadata.csv"


def save_signals(signals, sr, output_dir, log_melspectrogram=False):
    
    for i, signal in enumerate(signals):
        file_name = output_dir / f"{i}.wav"
        soundfile.write(file_name, signal, sr)

        if log_melspectrogram:
            file_name = output_dir / f"{i}.png"
            S = librosa.feature.melspectrogram(
                    signal, SR, n_fft=N_FFT, n_mels=N_MELS)
            librosa.display.specshow(S, x_axis='time',
                         y_axis='mel', sr=SR)

            plt.savefig(file_name)

def save_experiment_metadata(experiment_configs, output_dir):
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
