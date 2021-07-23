import shutil

from neural_tdoa.utils.callbacks import make_callbacks
from neural_tdoa.metrics import Loss
from datasets.dataset import TdoaDataset
from neural_tdoa.model import TdoaCrnn10
from neural_tdoa.train import train


def run_experiment(dataset_configs):
    for case_name, dataset_config in dataset_configs.items():
        _train(dataset_config, case_name)


def _train(dataset_config, case_name):
    shutil.rmtree(dataset_config["training_dataset_dir"], ignore_errors=True)
    shutil.rmtree(dataset_config["validation_dataset_dir"], ignore_errors=True)

    model = TdoaCrnn10()
    
    dataset_train = TdoaDataset(dataset_config)
    dataset_val = TdoaDataset(dataset_config, is_validation=True)

    loss_function = Loss()

    train(model, loss_function, dataset_train, dataset_val,
          callbacks=make_callbacks(), batch_size=32)
