import os
import numpy as np
import random

from omegaconf.omegaconf import open_dict
from pathlib import Path
from tqdm import tqdm

from datasets.logger import save_dataset_metadata
from datasets.generate_microphone_samples import (
    generate_random_training_sample_config,
    generate_microphone_samples
)

def generate_dataset(dataset_config,
                     log_melspectrogram=False):
    
    n_samples = dataset_config["n_samples"]
    output_dir = dataset_config["dataset_dir"]

    output_dir = Path(output_dir)
    output_samples_dir = output_dir / "samples"
    os.makedirs(output_samples_dir, exist_ok=True)

    if dataset_config["anechoic_samples_dir"]:
        path = Path(dataset_config["anechoic_samples_dir"])
        with open_dict(dataset_config):
            dataset_config["anechoic_samples"] = [
                str(p) for p in path.rglob("*.wav")
            ]
        

    random.seed(dataset_config["random_seed"])
    np.random.seed(dataset_config["random_seed"])
    
    training_sample_configs = []
    for num_sample in tqdm(range(n_samples)):
        training_sample_config = generate_random_training_sample_config(dataset_config)
        
        training_sample_config["signals_dir"] = output_samples_dir / str(num_sample)
        training_sample_configs.append(training_sample_config)

        generate_microphone_samples(training_sample_config,
                         log_melspectrogram=log_melspectrogram)

    save_dataset_metadata(training_sample_configs, output_dir)


if __name__ == "__main__":
    generate_dataset()
