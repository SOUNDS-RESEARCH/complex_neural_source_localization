import pandas as pd
import torch
import torchaudio
import ast

from pathlib import Path

SR = 16000
N_MICROPHONE_SECONDS = 1
N_MICS = 4


class SydraDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_dir,
                 sr=SR,
                 n_microphone_seconds=N_MICROPHONE_SECONDS,
                 n_mics=N_MICS):

        self.sr = sr
        self.n_microphone_seconds = n_microphone_seconds
        self.sample_duration = self.sr*self.n_microphone_seconds
        self.n_mics = n_mics

        self.df = _load_dataframe(dataset_dir)

    def __getitem__(self, index):
        if index >= len(self): raise IndexError
        
        sample_metadata = self.df.loc[index]

        x = torch.vstack([
            torchaudio.load(sample_metadata["signals_dir"] / f"{mic_idx}.wav")[0]
            for mic_idx in range(self.n_mics)
        ])

        x = x[:, :self.sample_duration]

        y = sample_metadata.to_dict()
        y = _desserialize_lists_within_dict(y)

        return (x, y)

    def __len__(self):
        return self.df.shape[0]


def _load_dataframe(dataset_dir):
    def _load(dataset_dir):
        dataset_dir = Path(dataset_dir)
        df = pd.read_csv(dataset_dir / "metadata.csv")
        
        df["signals_dir"] = df["signals_dir"].apply(
            lambda x: dataset_dir / x)

        return df
    
    if type(dataset_dir) in [str, Path]:
        df = _load(dataset_dir)
    else: # Multiple datasets
        dfs = [_load(d) for d in dataset_dir]
        df = pd.concat(dfs, ignore_index=True)

    return df


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
