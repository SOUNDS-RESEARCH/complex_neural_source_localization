
import shutil

from neural_tdoa.metrics import Loss
from datasets.dataset import TdoaDataset

from neural_tdoa.model import TdoaCrnn10
from neural_tdoa.train import train
from experiments.settings import BASE_EXPERIMENT_CONFIG


def run_experiment(experiment_configs=[BASE_EXPERIMENT_CONFIG]):
    for experiment_config in experiment_configs:
        _train(experiment_config)


def _train(experiment_config):
    dataset_config = experiment_config["dataset_config"]
    model_config = experiment_config["model_config"]
    training_config = experiment_config["training_config"]
    log_dir = experiment_config["log_dir"]

    _clean_dataset_dirs(dataset_config)

    model = TdoaCrnn10(feature_type=model_config["feature_type"])
    loss_function = Loss()
    
    dataset_train = TdoaDataset(dataset_config)
    dataset_val = TdoaDataset(dataset_config, is_validation=True)

    train(model, loss_function, dataset_train, dataset_val,
          log_dir=log_dir, training_config=training_config)


def _clean_dataset_dirs(dataset_config):
    # If there are files at these directories, datasets won't be regenerated
    shutil.rmtree(dataset_config["training_dataset_dir"], ignore_errors=True)
    shutil.rmtree(dataset_config["validation_dataset_dir"], ignore_errors=True)


if __name__ == "__main__":
    experiment_config = BASE_EXPERIMENT_CONFIG
    experiment_config["training_config"]["num_epochs"] = 1
    experiment_config["model_config"]["feature_type"] = "stft_magnitude"
    results = run_experiment([experiment_config])