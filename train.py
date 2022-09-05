import hydra

from omegaconf import DictConfig

from complex_neural_source_localization.datasets import create_dataloaders
from complex_neural_source_localization.trainer import DOACNetTrainer


@hydra.main(config_path="config", config_name="config", version_base=None)
def train(config: DictConfig):
    """Runs the training procedure using Pytorch lightning
    And tests the model with the best validation score against the test dataset. 

    Args:
        config (DictConfig): Configuration automatically loaded by Hydra.
                                        See the config/ directory for the configuration
    """

    dataset_train, dataset_val, dataset_test = create_dataloaders(config)
    
    trainer = DOACNetTrainer(config)

    trainer.fit(dataset_train, val_dataloaders=dataset_val)
    trainer.test(dataset_test)


if __name__ == "__main__":
    train()
