import librosa.display
import matplotlib.pyplot as plt
import soundfile
import pandas as pd

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
    output_keys = [
        "source_x", "source_y", "tdoa", "normalized_tdoa", "signals_dir"
    ]
    
    def filter_keys(training_sample_config):
        output_dict = {
            key: value for key, value in training_sample_config.items()
            if key in output_keys
        }

        return output_dict
    
    output_dicts = [filter_keys(exp) for exp in training_sample_configs]

    df = pd.DataFrame(output_dicts)
    df.to_csv(output_dir / METADATA_FILENAME)
