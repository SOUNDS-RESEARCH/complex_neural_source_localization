import os
import random
import numpy as np

from omegaconf.omegaconf import open_dict
from pathlib import Path
from tqdm import tqdm

from datasets.acoustics_simulator import simulate_microphone_signals
from datasets.logger import save_dataset_metadata, save_signals
from datasets.random import generate_sample_config


def generate_datasets(dataset_configs):
    for dataset_config in dataset_configs:
        generate_dataset(dataset_config)


def generate_dataset(dataset_config):
    
    n_samples = dataset_config["n_samples"]
    output_dir = dataset_config["dataset_dir"]

    output_dir = Path(output_dir)
    output_samples_dir = output_dir / "samples"
    os.makedirs(output_samples_dir, exist_ok=True)

    if dataset_config["speech_signals_dir"]:
        path = Path(dataset_config["speech_signals_dir"])
        with open_dict(dataset_config):
            dataset_config["speech_samples"] = [
                str(p) for p in path.rglob("*.wav")
            ]

    random.seed(dataset_config["random_seed"])
    np.random.seed(dataset_config["random_seed"])
    
    training_sample_configs = []
    for num_sample in tqdm(range(n_samples)):
        training_sample_config = generate_sample_config(dataset_config)
        
        training_sample_config["signals_dir"] = output_samples_dir / str(num_sample)
        training_sample_configs.append(training_sample_config)

        generate_dataset_sample(training_sample_config)

    save_dataset_metadata(training_sample_configs, output_dir)


def generate_dataset_sample(config, log_melspectrogram=False):

    output_signals = simulate_microphone_signals(config)
    output_signals = _trim_microphone_signals(output_signals, config)

    os.makedirs(config["signals_dir"], exist_ok=True)
    save_signals(output_signals,
                 config["sr"],
                 config["signals_dir"],
                 log_melspectrogram)


# TODO: Add this option to pyroomasync
def _trim_microphone_signals(signals, config):
    """Trim beginning and end of signals not to have a silence in the beginning,
    which might make the delay detection too easy 
    """

    n_microphone_samples = int(config["n_microphone_seconds"]*config["sr"])
    # The delay simulation consists in 
    if config["trim_beginning"]:
        # Number of maximum delayed samples
        max_delay = int(max(config["mic_delays"])*config["sr"])
        signals = signals[:, max_delay:]

    signals = signals[:, :n_microphone_samples]

    return signals
