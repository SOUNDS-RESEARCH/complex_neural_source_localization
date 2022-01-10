import librosa
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import Dataset


class DCASE2019Task3Dataset(Dataset):
    def __init__(self, dataset_config, mode="train"):
        self.config = dataset_config
        self.sr = dataset_config["sr"]
        self.sample_duration_in_seconds = dataset_config["sample_duration_in_seconds"]
        self.sample_duration = self.sr*self.sample_duration_in_seconds
        self.num_mics = dataset_config["num_mics"]

        if mode == "train":
            annotations_path = dataset_config["train_csv_path"]
            self.wav_path = Path(dataset_config["train_wav_path"])
        elif mode == "validation":
            annotations_path = dataset_config["validation_csv_path"]
            self.wav_path = Path(dataset_config["validation_wav_path"])
        elif mode == "test":
            annotations_path = dataset_config["test_csv_path"]
            self.wav_path = Path(dataset_config["test_wav_path"])
        else:
            raise ValueError(f"Dataset mode {mode} is invalid")

        self.df = pd.read_csv(annotations_path)

    def __getitem__(self, index):
        annotation = self.df.iloc[index]
        wav_file_path = self.wav_path / (annotation["file_name"] + ".wav")

        signal, _ = librosa.load(wav_file_path,
                                 sr=self.sr, mono=False, dtype=np.float32)
        padded_signal = np.zeros((self.num_mics, self.sample_duration))
        padded_signal[:, :signal.shape[1]] = signal
        signal = padded_signal
        # TODO: Remove samples from the dataset which are incomplete

        azimuth_in_degrees = annotation["azi"]
        azimuth_in_radians = np.deg2rad(azimuth_in_degrees)
        azimuth_complex_point = torch.complex(
            torch.Tensor([np.cos(azimuth_in_radians)]),
            torch.Tensor([np.sin(azimuth_in_radians)])
        )
        azimuth_2d_point = torch.Tensor([
            torch.Tensor([np.cos(azimuth_in_radians)]),
            torch.Tensor([np.sin(azimuth_in_radians)])
        ])

        return (
            torch.Tensor(signal),
            {
                "azimuth_complex_point": azimuth_complex_point,
                "azimuth_2d_point": azimuth_2d_point,
                "start_time": torch.Tensor([annotation["start_time"]]),
                "end_time": torch.Tensor([annotation["end_time"]])
            }
        )

    def __len__(self):
        return self.df.shape[0]


if __name__ == "__main__":

    dataset = DCASE2019Task3Dataset(
                "/Users/ezajlerg/datasets/dcase_2019_task_3/annotations.csv",
                "/Users/ezajlerg/datasets/dcase_2019_task_3/mic_dev_splitted")

