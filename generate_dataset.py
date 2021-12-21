from datasets.generate_dataset import generate_datasets
import hydra

from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="config")
def main(config: DictConfig):
    """Generate dataset according to config in the config/ folder

    Args:
        config (DictConfig): Configuration automatically loaded by Hydra.
                                        See the config/ directory for the configuration
    """

    generate_datasets([
        config["training_dataset"],
        config["validation_dataset"],
        config["test_dataset"]
    ])


if __name__ == "__main__":
    main()
