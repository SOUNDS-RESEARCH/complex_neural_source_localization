import os

from omegaconf.omegaconf import open_dict
from pathlib import Path
from tqdm import tqdm

from datasets.logger import save_dataset_metadata
from datasets.generate_dataset_sample import (
    generate_dataset_sample_config,
    generate_dataset_sample
)


def generate_datasets(dataset_configs):
    for dataset_config in tqdm(dataset_configs):
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

    # random.seed(dataset_config["random_seed"])
    # np.random.seed(dataset_config["random_seed"])
    
    training_sample_configs = []
    for num_sample in tqdm(range(n_samples)):
        training_sample_config = generate_dataset_sample_config(dataset_config)
        
        training_sample_config["signals_dir"] = output_samples_dir / str(num_sample)
        training_sample_configs.append(training_sample_config)

        generate_dataset_sample(training_sample_config)

    save_dataset_metadata(training_sample_configs, output_dir)


if __name__ == "__main__":
    generate_dataset()
