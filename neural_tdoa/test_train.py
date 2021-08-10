import shutil

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


from datasets.dataset import TdoaDataset
from neural_tdoa.train import train
from neural_tdoa.metrics import Loss

from neural_tdoa.model import TdoaCrnn10


def test_train(regenerate_datasets=False):
    config = _load_config()
    _setup(config["dataset"], regenerate_datasets)


    model = TdoaCrnn10(config["model"], config["dataset"])
    
    dataset_train = TdoaDataset(config["dataset"])

    dataset_val = TdoaDataset(config["dataset"], is_validation=True)

    loss_function = Loss()

    train(model, loss_function, dataset_train, dataset_val)


def _setup(dataset_config, regenerate_datasets):
    if regenerate_datasets:
        shutil.rmtree(dataset_config["training_dataset_dir"], ignore_errors=True)
        shutil.rmtree(dataset_config["validation_dataset_dir"], ignore_errors=True)


def _load_config():
    GlobalHydra.instance().clear()
    temp_dataset_path = "tests/temp/dataset"
    shutil.rmtree(temp_dataset_path, ignore_errors=True)

    initialize(config_path="../config", job_name="test_app")
    cfg = compose(config_name="config")
    cfg["dataset"]["training_dataset_dir"] = "tests/temp/train_dataset_dir"
    cfg["dataset"]["validation_dataset_dir"] = "tests/temp/validation_dataset_dir"
    cfg["dataset"]["n_training_samples"] = 350
    cfg["dataset"]["n_validation_samples"] = 150
    cfg["model"]["feature_type"] = "stft_magnitude"

    return cfg


if __name__ == "__main__":
    test_train(False)
