import os
import torch
import torchaudio

from pathlib import Path

from datasets.settings import SR, TEMPDIR
from datasets.dcase_2021.dcase_2021_task3_annotation import (
    load_csv_as_dataframe,
    load_event_from_csv
)

DEFAULT_DATASET_DIR = Path(TEMPDIR) / "dcase_2021_task3"
DEFAULT_SIGNALS_DIR = DEFAULT_DATASET_DIR / "mic_dev"
DEFAULT_ANNOTATIONS_DIR = DEFAULT_DATASET_DIR / "metadata_dev"


class DCASE2021Task3Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 file_paths=[], sr=SR, mode="train",
                 signals_dir=DEFAULT_SIGNALS_DIR,
                 annotations_dir=DEFAULT_ANNOTATIONS_DIR,
                 annotation_mode="file_path"):

        self.sr = sr
        self.annotation_mode = annotation_mode
        self.file_paths = file_paths if file_paths else _load_file_paths(
            mode, signals_dir, annotations_dir)

    def __getitem__(self, index):
        x_file_path, y_file_path = self.file_paths[index]

        signal, _ = torchaudio.load(x_file_path)

        if self.annotation_mode == "event":
            targets = load_event_from_csv(y_file_path)
        elif self.annotation_mode == "load":
            targets = load_csv_as_dataframe(y_file_path)
        elif self.annotation_mode == "file_path":
            targets = y_file_path

        return {
            "signal": signal,
            "targets": targets
        }

    def __len__(self):
        return len(self.file_paths)


def _load_file_paths(mode="train",
                    signals_dir=DEFAULT_SIGNALS_DIR,
                    annotations_dir=DEFAULT_ANNOTATIONS_DIR):
    valid_modes = ["train", "test", "val"]
    signals_dir = Path(signals_dir)
    annotations_dir = Path(annotations_dir)

    if mode not in valid_modes:
        print(f"Invalid mode '{mode}'")
        print(f"Valid modes are: {valid_modes}")

    x_dir_path = signals_dir / f"dev-{mode}"
    y_dir_path = annotations_dir / f"dev-{mode}"
    file_names = os.listdir(x_dir_path)
    # The files have the same names, only different folders

    file_paths = [
        (
            x_dir_path / file_name,
            y_dir_path / file_name.replace(".wav", ".csv")
        )
        for file_name in file_names
    ]

    return file_paths
