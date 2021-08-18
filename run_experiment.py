import hydra
from omegaconf import DictConfig

from neural_tdoa.train import train_pl


@hydra.main(config_path="config", config_name="config")
def main(experiment_config: DictConfig):
    train_pl(experiment_config)


if __name__ == "__main__":
    main()
