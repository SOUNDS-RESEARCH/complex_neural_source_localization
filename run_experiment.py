import hydra
import os
import pandas as pd

from omegaconf import DictConfig
from pathlib import Path

from datasets.dataset import TdoaDataset
from neural_tdoa.metrics import Loss
from neural_tdoa.model import TdoaCrnn10
from neural_tdoa.train import train, train_pl


@hydra.main(config_path="config", config_name="config")
def main(experiment_config: DictConfig):
    train_pl(experiment_config)



@hydra.main(config_path="config", config_name="config")
def run_experiment(experiment_config: DictConfig):
    dataset_config = experiment_config["dataset"]
    model_config = experiment_config["model"]
    training_config = experiment_config["training"]

    model = TdoaCrnn10(model_config, dataset_config)
    loss_function = Loss()
    
    dataset_train = TdoaDataset(dataset_config)
    dataset_val = TdoaDataset(dataset_config, is_validation=True)

    log_dir = Path(os.getcwd())
    train(training_config, model, loss_function,
          dataset_train, dataset_val,
          log_dir=log_dir)

    train_results = pd.read_csv(log_dir / "train.csv")
    validation_results = pd.read_csv(log_dir / "valid.csv")

    return train_results, validation_results


if __name__ == "__main__":
    main()
