from omegaconf import OmegaConf
import torch

from torch.optim.lr_scheduler import MultiStepLR

from complex_neural_source_localization.model import DOACNet
from complex_neural_source_localization.loss import Loss, PitLoss
from complex_neural_source_localization.utils.base_trainer import (
    BaseTrainer, BaseLightningModule
)


class DOACNetTrainer(BaseTrainer):
    def __init__(self, config):
        lightning_module = DOACNetLightniningModule(config)
        super().__init__(lightning_module,
                     config["training"]["n_epochs"])
    
    def fit(self, train_dataloaders, val_dataloaders=None):
        super().fit(self._lightning_module, train_dataloaders,
                    val_dataloaders=val_dataloaders)
    
    def test(self, test_dataloaders):
        super().test(self._lightning_module, test_dataloaders, ckpt_path="best")


class DOACNetLightniningModule(BaseLightningModule):
    """This class abstracts the
       training/validation/testing procedures
       used for training a DOACNet
    """
    def __init__(self, config):
        config = OmegaConf.to_container(config)
        self.config = config
        
        n_sources = self.config["dataset"]["n_max_sources"]


        # model_config = config["model"]
        # model = DOACNet(
        #     conv_config=model_config["conv_layers"],
        #     fc_layer_dropout_rate=model_config["fc_layer_dropout_rate"],
        #     n_sources=n_sources,
        #     pool_size=model_config["pool_size"],
        #     pool_type=model_config["pool_type"],
        #     feature_type=model_config["feature_type"],
        #     activation=model_config["activation"],
        #     complex_to_real_mode=model_config["complex_to_real_mode"],
        #     kernel_size=model_config["kernel_size"],
        #     use_complex_rnn=model_config["use_complex_rnn"]
        # )

        model = DOACNet(n_sources=n_sources, **config["model"])


        if n_sources == 2:
            loss = PitLoss(self.config["model"]["loss"])
        elif n_sources == 1:
            loss = Loss(self.config["model"]["loss"])
        else:
            raise NotImplementedError(
                "Loss function was only implemented for one and 2 sources."
            )

        super().__init__(model, loss)

    def configure_optimizers(self):
        lr = self.config["training"]["learning_rate"]
        decay_step = self.config["training"]["learning_rate_decay_steps"]
        decay_value = self.config["training"]["learning_rate_decay_values"]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, decay_step, decay_value)

        return [optimizer], [scheduler]
