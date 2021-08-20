from neural_tdoa.utils.load_config import load_config
import os
import pandas as pd
import torch
import torchaudio

from pathlib import Path

from datasets.generate_dataset import generate_dataset


class TdoaDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_config=None):
        
        if dataset_config is None:
            dataset_config = load_config("training_dataset")

        self.sr = dataset_config["base_sampling_rate"]
        self.sample_duration_in_secs = dataset_config["sample_duration_in_secs"]
        self.sample_duration = self.sr*self.sample_duration_in_secs
        self.n_mics = len(dataset_config["mic_coordinates"])

        dataset_dir = dataset_config["dataset_dir"]

        if not os.path.exists(dataset_dir):
            generate_dataset(dataset_config)

        self.df = pd.read_csv(Path(dataset_dir) / "metadata.csv") 

    def __getitem__(self, index):
        sample_metadata = self.df.loc[index]
        signals_dir = Path(sample_metadata["signals_dir"])

        x = torch.vstack([
            torchaudio.load(signals_dir / f"{mic_idx}.wav")[0]
            for mic_idx in range(self.n_mics)
        ])

        x = x[:, :self.sample_duration]
        y = torch.Tensor([sample_metadata["normalized_tdoa"]])

        return (x, y)

    def __len__(self):
        return self.df.shape[0]
