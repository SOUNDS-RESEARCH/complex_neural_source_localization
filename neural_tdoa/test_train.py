import shutil

from datasets.settings import BASE_DATASET_CONFIG
from datasets.dataset import TdoaDataset
from neural_tdoa.train import train
from neural_tdoa.metrics import Loss

from neural_tdoa.model import TdoaCrnn10

DATASET_CONFIG = BASE_DATASET_CONFIG
DATASET_CONFIG["n_training_samples"] = 350
DATASET_CONFIG["n_validation_samples"] = 150

DATASET_CONFIG["training_dataset_dir"] = "tests/temp/train_dataset_dir"
DATASET_CONFIG["validation_dataset_dir"] = "tests/temp/validation_dataset_dir"


def test_train(regenerate_datasets=False):
    _setup(regenerate_datasets)

    model = TdoaCrnn10()
    
    dataset_train = TdoaDataset(DATASET_CONFIG)

    dataset_val = TdoaDataset(DATASET_CONFIG, is_validation=True)

    loss_function = Loss()

    train(model, loss_function, dataset_train, dataset_val)


def _setup(regenerate_datasets):
    if regenerate_datasets:
        shutil.rmtree(DATASET_CONFIG["training_dataset_dir"], ignore_errors=True)
        shutil.rmtree(DATASET_CONFIG["validation_dataset_dir"], ignore_errors=True)

if __name__ == "__main__":
    test_train(False)
