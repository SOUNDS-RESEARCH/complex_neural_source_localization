import torch

from .dcase_2019_task3_dataset import DCASE2019Task3Dataset
from .synthetic_ssl_dataset import SyntheticSSLDataset

DATASET_NAME_TO_CLASS_MAP = {
    "synthetic_ssl": SyntheticSSLDataset,
    "dcase_2019_task3": DCASE2019Task3Dataset
}


def create_dataloaders(config):

    dataset = DATASET_NAME_TO_CLASS_MAP[config["dataset"]["name"]]
    dataset_train = dataset(config["dataset"]["training_dataset_dir"])
    dataset_val = dataset(config["dataset"]["validation_dataset_dir"])
    dataset_test = dataset(config["dataset"]["test_dataset_dir"])

    batch_size = config["training"]["batch_size"]
    n_workers = config["training"]["n_workers"]

    dataloader_train = _create_torch_dataloader(dataset_train, batch_size,
                                                n_workers, shuffle=True)
    dataloader_val = _create_torch_dataloader(dataset_val, batch_size,
                                              n_workers)
    dataloader_test = _create_torch_dataloader(dataset_test, batch_size,
                                               n_workers)

    return dataloader_train, dataloader_val, dataloader_test


def _create_torch_dataloader(torch_dataset, batch_size, n_workers, shuffle=False):
    return torch.utils.data.DataLoader(torch_dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       pin_memory=True,
                                       drop_last=False,
                                       num_workers=n_workers)
