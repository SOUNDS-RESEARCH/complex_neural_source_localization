import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import ProgressBar
from torch.optim.lr_scheduler import MultiStepLR

from complex_neural_source_localization.model import Crnn10
from complex_neural_source_localization.loss import Loss, PitLoss


def create_trainer(training_config):
    checkpoint_callback = ModelCheckpoint(
                            monitor="validation_loss",
                            save_last=True,
                            filename='weights-{epoch:02d}-{validation_loss:.2f}',
                            save_weights_only=True
                          )

    gpus = 1 if torch.cuda.is_available() else 0

    progress_bar = CustomProgressBar()
    trainer = pl.Trainer(max_epochs=training_config["n_epochs"],
                         callbacks=[checkpoint_callback, progress_bar],
                         gpus=gpus)
    return trainer


class LitTdoaCrnn(pl.LightningModule):
    """Pytorch lightning class used to abstract the
        training/validation/testing procedure of the TdoaCrnn neural network
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.target_key = config["model"]["target"]
        
        n_sources = self.config["dataset"]["n_max_sources"]

        self.model = Crnn10(conv_config=config["model"]["conv_layers"],
                            last_layer_dropout_rate=config["model"]["last_layer_dropout_rate"],
                            n_sources=n_sources
                            )
        
        if n_sources == 2:
            self.loss = PitLoss(self.config["model"]["loss"])
        else:
            self.loss = Loss(self.config["model"]["loss"])


    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.model(x)
        loss = self.loss(predictions, y)

        output_dict = {
            "loss": loss
        }

        return output_dict
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.model(x)

        loss = self.loss(predictions, y)
        output_dict = {
            "loss": loss
        }

        return output_dict

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("loss", avg_loss, on_epoch=True)
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("validation_loss", avg_loss, prog_bar=True)
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("test_loss", avg_loss, on_step=False)

    def configure_optimizers(self):
        lr = self.config["training"]["learning_rate"]
        decay_step = self.config["training"]["learning_rate_decay_steps"]
        decay_value = self.config["training"]["learning_rate_decay_values"]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, decay_step, decay_value)

        return [optimizer], [scheduler]


class CustomProgressBar(ProgressBar):
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items