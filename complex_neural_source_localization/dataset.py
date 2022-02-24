import librosa
import numpy as np
import pandas as pd
import torch

from math import isnan
from pathlib import Path
from torch.utils.data import Dataset


class DCASE2019Task3Dataset(Dataset):
    def __init__(self, dataset_config, mode="train"):
        self.config = dataset_config
        self.sr = dataset_config["sr"]
        self.sample_duration_in_seconds = dataset_config["sample_duration_in_seconds"]
        self.sample_duration = self.sr*self.sample_duration_in_seconds
        self.num_mics = dataset_config["num_mics"]
        self.n_max_sources = dataset_config["n_max_sources"]

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

        azimuth_in_radians = np.deg2rad(annotation["azi"])
        azimuth_2d_point = _angle_to_point(azimuth_in_radians)

        y = {
                "azimuth_2d_point": azimuth_2d_point,
                # "start_time": torch.Tensor([annotation["start_time"]]),
                # "end_time": torch.Tensor([annotation["end_time"]])
        }

        # If multi-source
        if self.n_max_sources == 2:
            # If there is a second source, use it. Else, copy first source
            if not isnan(annotation["azi_2"]):
                azi = annotation["azi_2"]
            else:
                azi = annotation["azi"]
            azimuth_in_radians_2 = np.deg2rad(azi)
            azimuth_2d_point_2 = _angle_to_point(azimuth_in_radians_2)
            y["azimuth_2d_point_2"] = azimuth_2d_point_2


        return (torch.Tensor(signal), y)

    def __len__(self):
        return self.df.shape[0]


def create_dataloaders(config):
    dataset_train = DCASE2019Task3Dataset(config["dataset"], mode="train")
    dataset_val = DCASE2019Task3Dataset(config["dataset"], mode="validation")
    dataset_test = DCASE2019Task3Dataset(config["dataset"], mode="test")

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    dataloader_train = _create_torch_dataloader(dataset_train, batch_size, num_workers)
    dataloader_val = _create_torch_dataloader(dataset_val, batch_size, num_workers)
    dataloader_test = _create_torch_dataloader(dataset_test, batch_size, num_workers)

    return dataloader_train, dataloader_val, dataloader_test


def _angle_to_point(angle):
    return torch.Tensor([
        torch.Tensor([np.cos(angle)]),
        torch.Tensor([np.sin(angle)])
    ])


def _create_torch_dataloader(torch_dataset, batch_size, num_workers):
    return torch.utils.data.DataLoader(torch_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       drop_last=False,
                                       num_workers=num_workers)
