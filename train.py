import hydra
import torch

from omegaconf import DictConfig

from datasets.dcase_2019_task3_dataset import DCASE2019Task3Dataset
from neural_tdoa.trainer import LitTdoaCrnn, create_trainer


@hydra.main(config_path="config", config_name="config")
def main(config: DictConfig):
    """Runs the training procedure using Pytorch lightning
    And tests the model with the best validation score against the test dataset. 

    Args:
        config (DictConfig): Configuration automatically loaded by Hydra.
                                        See the config/ directory for the configuration
    """

    model = LitTdoaCrnn(config)
    dataset_train, dataset_val, dataset_test = _create_dataloaders(config)
    
    trainer = create_trainer(config["training"])
    trainer.fit(model, dataset_train, val_dataloaders=dataset_val)
    trainer.test(model, dataset_test, ckpt_path="best")


def _create_dataloaders(config):
    dataset_train = DCASE2019Task3Dataset(config["dcase_2019_task3_dataset"], mode="train")
    dataset_val = DCASE2019Task3Dataset(config["dcase_2019_task3_dataset"], mode="validation")
    dataset_test = DCASE2019Task3Dataset(config["dcase_2019_task3_dataset"], mode="test")

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    dataloader_train = _create_torch_dataloader(dataset_train, batch_size, num_workers)
    dataloader_val = _create_torch_dataloader(dataset_val, batch_size, num_workers)
    dataloader_test = _create_torch_dataloader(dataset_test, batch_size, num_workers)

    return dataloader_train, dataloader_val, dataloader_test


def _create_torch_dataloader(torch_dataset, batch_size, num_workers):
    return torch.utils.data.DataLoader(torch_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       drop_last=False,
                                       num_workers=num_workers)


if __name__ == "__main__":
    main()
