import os
import pandas as pd
import torch
import torchaudio

from pathlib import Path

from datasets.generate_dataset import generate_dataset
from datasets.settings import BASE_DATASET_CONFIG


class TdoaDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_config=BASE_DATASET_CONFIG,
                 is_validation=False):

        self.sr = dataset_config["base_sampling_rate"]
        self.sample_duration_in_secs = dataset_config["sample_duration_in_secs"]
        self.sample_duration = self.sr*self.sample_duration_in_secs
        self.n_mics = len(dataset_config["mic_coordinates"])
        
        if is_validation:
            dataset_dir = dataset_config["training_dataset_dir"]
        else:
            dataset_dir = dataset_config["validation_dataset_dir"]

        if not os.path.exists(dataset_dir):
            generate_dataset(dataset_config, is_validation)

        self.df = pd.read_csv(Path(dataset_dir) / "metadata.csv") 

    def __getitem__(self, index):

        sample_metadata = self.df.loc[index]
        signals_dir = Path(sample_metadata["signals_dir"])

        x = torch.vstack([
            torchaudio.load(signals_dir / f"{mic_idx}.wav")[0]
            for mic_idx in range(self.n_mics)
        ])

        # Note: Truncating to 2 seconds: losing reverberated "tail" at the end.
        # Ask Patrick if its a problem
        x = x[:, :self.sample_duration]

        y = torch.Tensor([sample_metadata["normalized_tdoa"]])

        return {
            "signals": x,
            "targets": y
        }

    def __len__(self):
        return self.df.shape[0]
