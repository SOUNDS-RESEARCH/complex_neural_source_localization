from complex_neural_source_localization.utils.load_config import load_config
import os
import pandas as pd
import torch
import torchaudio
import ast

from pathlib import Path

from datasets.generate_dataset import generate_dataset


class TdoaDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_config=None):

        if dataset_config is None:
            dataset_config = load_config("training_dataset")
        self.config = dataset_config

        self.sr = dataset_config["base_sampling_rate"]
        self.n_microphone_seconds = dataset_config["n_microphone_seconds"]
        self.sample_duration = self.sr*self.n_microphone_seconds
        self.n_mics = dataset_config["n_mics"]
        self.dataset_dir = Path(dataset_config["dataset_dir"])

        if not os.path.exists(self.dataset_dir):
            print("dataset directory does not exist. Generating new dataset")
            generate_dataset(dataset_config)

        self.df = pd.read_csv(self.dataset_dir / "metadata.csv") 

    def __getitem__(self, index):
        if index >= len(self): raise IndexError
        
        sample_metadata = self.df.loc[index]
        signals_dir = self.dataset_dir / sample_metadata["signals_dir"]

        x = torch.vstack([
            torchaudio.load(signals_dir / f"{mic_idx}.wav")[0]
            for mic_idx in range(self.n_mics)
        ])

        x = x[:, :self.sample_duration]

        y = sample_metadata.to_dict()
        y = _desserialize_lists_within_dict(y)
        y["normalized_tdoa"] = torch.Tensor([y["normalized_tdoa"]])
        y["azimuth_in_radians"] = torch.Tensor([y["azimuth_in_radians"]])

        y["azimuth_complex_point"] = torch.complex( # Transform 2d unit vector into complex number
            y["azimuth_complex_point"][0],
            y["azimuth_complex_point"][1]).unsqueeze(0)

        return (x, y)

    def __len__(self):
        return self.df.shape[0]


def _desserialize_lists_within_dict(d):
    """Lists were saved in pandas as strings.
       This small utility function transforms them into lists again.
    """
    new_d = {}
    for key, value in d.items():
        if type(value) == str:
            try:
                new_value = ast.literal_eval(value)
                new_d[key] = torch.Tensor(new_value)
            except (SyntaxError, ValueError):
                new_d[key] = value
        else:
            new_d[key] = value
    return new_d
