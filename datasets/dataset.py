import os
import pandas as pd
import torch
import torchaudio

from pathlib import Path

from datasets.generate_dataset import generate_dataset
from datasets.settings import (
    N_SAMPLES, SR, DEFAULT_OUTPUT_DATASET_DIR, N_MICS, SAMPLE_DURATION_IN_SECS
)


class TdoaDataset(torch.utils.data.Dataset):
    def __init__(self,
                 sr=SR,
                 n_samples=N_SAMPLES,
                 dataset_dir=DEFAULT_OUTPUT_DATASET_DIR,
                 sample_duration_in_secs=SAMPLE_DURATION_IN_SECS,
                 n_mics=N_MICS):

        self.sr = sr
        self.sample_duration_in_secs = sample_duration_in_secs
        self.sample_duration = sr*sample_duration_in_secs
        self.n_mics = n_mics

        if not os.path.exists(dataset_dir):
            generate_dataset(dataset_dir,
                             num_samples=n_samples,
                             sample_duration_in_secs=sample_duration_in_secs)

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
