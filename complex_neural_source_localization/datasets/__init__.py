import torch

from .dcase_2019_task3_dataset import DCASE2019Task3Dataset


def create_dataloaders(config):
    dataset_train = DCASE2019Task3Dataset(config["dataset"])
    dataset_val = DCASE2019Task3Dataset(config["dataset"])
    dataset_test = DCASE2019Task3Dataset(config["dataset"])

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
