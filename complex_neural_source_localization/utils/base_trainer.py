import pickle
import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from complex_neural_source_localization.utils.model_utilities import (
    merge_list_of_dicts 
)
from complex_neural_source_localization.utils.logging import (
    ConvolutionalFeatureMapLogger 
)


class BaseTrainer(pl.Trainer):
    def __init__(self, lightning_module, n_epochs):

        gpus = 1 if torch.cuda.is_available() else 0

        class CustomProgressBar(TQDMProgressBar):
            def get_metrics(self, trainer, model):
                # don't show the version number
                items = super().get_metrics(trainer, model)
                items.pop("v_num", None)
                return items
        progress_bar = CustomProgressBar()

        checkpoint_callback = ModelCheckpoint(
                        monitor="validation_loss",
                        save_last=True,
                        filename='weights-{epoch:02d}-{validation_loss:.2f}',
                        save_weights_only=True
                        )

        super().__init__(max_epochs=n_epochs,
                         callbacks=[checkpoint_callback, progress_bar],
                         gpus=gpus)
        
        self._lightning_module = lightning_module


class BaseLightningModule(pl.LightningModule):
    """Class which abstracts interactions with Hydra
    and basic training/testing/validation conventions
    """

    def __init__(self, model, loss,
                 log_convolutional_feature_maps=True,
                 log_step=50):
        super().__init__()

        self.is_cuda_available = torch.cuda.is_available()

        self.model = model
        self.loss = loss

        self.log_convolutional_feature_maps = log_convolutional_feature_maps
        if log_convolutional_feature_maps:
            self.convolutional_feature_map_logger = ConvolutionalFeatureMapLogger(
                                                        model, self)

        self.log_step = log_step

    def _step(self, batch, batch_idx, log_model_output=False,
              log_labels=False, log_conv_feature_maps=False):

        x, y = batch

        # 1. Compute model output and loss
        output = self.model(x)
        loss = self.loss(output, y, mean_reduce=False)

        output_dict = {
            "loss_vector": loss
        }

        # 2. Log model output
        if log_model_output:
            output_dict["model_output"] = output
        # 3. Log ground truth labels
        if log_labels:
            output_dict.update(y)

        # 4. Log feature maps of convolutional layers
        if log_conv_feature_maps and batch_idx%self.log_step == 0:
            image = self.convolutional_feature_map_logger.log()
            
            self.convolutional_feature_map_logger.log()

        output_dict["loss"] = output_dict["loss_vector"].mean()
        output_dict["loss_vector"] = output_dict["loss_vector"].detach()

        return output_dict

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)
  
    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx,
                          log_model_output=True, log_labels=True,
                          log_conv_feature_maps=self.log_convolutional_feature_maps)
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx,
                          log_model_output=True, log_labels=True,
                          log_conv_feature_maps=self.log_convolutional_feature_maps)
    
    def _epoch_end(self, outputs, epoch_type="train", save_pickle=False):
        # 1. Compute epoch metrics
        outputs = merge_list_of_dicts(outputs)
        epoch_stats = {
            f"{epoch_type}_loss": outputs["loss"].mean(),
            f"{epoch_type}_std": outputs["loss"].std()
        }

        # 2. Log epoch metrics
        for key, value in epoch_stats.items():
            self.log(key, value, on_epoch=True, prog_bar=True)

        # 3. Save complete epoch data on pickle
        if save_pickle:
            pickle_filename = f"{epoch_type}.pickle"
            with open(pickle_filename, "wb") as f:
                pickle.dump(outputs, f)

        return epoch_stats
    
    def training_epoch_end(self, outputs):
        self._epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, epoch_type="validation")

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, epoch_type="test", save_pickle=True)
    def forward(self, x):
        return self.model(x)
        
    def fit(self, dataset_train, dataset_val):
        super().fit(self.model, dataset_train, val_dataloaders=dataset_val)

    def test(self, dataset_test):
        super().test(self.model, dataset_test, ckpt_path="best")
