from datasets.generate_dataset import generate_dataset, generate_datasets
import hydra
import shutil
import torch

from omegaconf import DictConfig

from datasets.dataset import TdoaDataset
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

    if config["training"]["delete_datasets_after_training"]:
        shutil.rmtree(config["training_dataset"]["dataset_dir"])
        shutil.rmtree(config["validation_dataset"]["dataset_dir"])
        shutil.rmtree(config["test_dataset"]["dataset_dir"])


def _create_dataloaders(config):
    generate_datasets([
        config["training_dataset"],
        config["validation_dataset"],
        config["test_dataset"]
    ])

    dataset_train = TdoaDataset(config["training_dataset"])
    dataset_val = TdoaDataset(config["validation_dataset"])
    dataset_test = TdoaDataset(config["test_dataset"])

    batch_size = config["training"]["batch_size"]
    dataloader_train = _create_torch_dataloader(dataset_train, batch_size)
    dataloader_val = _create_torch_dataloader(dataset_val, batch_size)
    dataloader_test = _create_torch_dataloader(dataset_test, batch_size)

    return dataloader_train, dataloader_val, dataloader_test


def _create_torch_dataloader(torch_dataset, batch_size):
    return torch.utils.data.DataLoader(torch_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       drop_last=False,
                                       num_workers=2)


if __name__ == "__main__":
    main()
