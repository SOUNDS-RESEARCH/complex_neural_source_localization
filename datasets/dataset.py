from neural_tdoa.utils.load_config import load_config
import os
import pandas as pd
import torch
import torchaudio
import ast

from pathlib import Path

from datasets.generate_dataset import generate_dataset


class TdoaDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_config=None,
                 include_metadata=True):

        if dataset_config is None:
            dataset_config = load_config("training_dataset")

        self.sr = dataset_config["base_sampling_rate"]
        self.sample_duration_in_secs = dataset_config["sample_duration_in_secs"]
        self.sample_duration = self.sr*self.sample_duration_in_secs
        self.n_mics = dataset_config["n_mics"]
        self.include_metadata = include_metadata
        dataset_dir = dataset_config["dataset_dir"]

        if not os.path.exists(dataset_dir):
            generate_dataset(dataset_config)

        self.df = pd.read_csv(Path(dataset_dir) / "metadata.csv") 

    def __getitem__(self, index):
        if index >= len(self): raise IndexError
        
        sample_metadata = self.df.loc[index]
        signals_dir = Path(sample_metadata["signals_dir"])

        x = torch.vstack([
            torchaudio.load(signals_dir / f"{mic_idx}.wav")[0]
            for mic_idx in range(self.n_mics)
        ])

        x = x[:, :self.sample_duration]
        y = torch.Tensor([sample_metadata["normalized_tdoa"]])

        if self.include_metadata:
            metadata_dict = sample_metadata.to_dict()
            metadata_dict = _desserialize_lists_within_dict(metadata_dict)
            metadata_dict["y"] = y
            y = metadata_dict
            
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
