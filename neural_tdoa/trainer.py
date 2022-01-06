import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import ModelCheckpoint

from neural_tdoa.crnns import Crnn10
from neural_tdoa.loss import Loss


def create_trainer(training_config):
    checkpoint_callback = ModelCheckpoint(
                            monitor="validation_loss",
                            save_last=True,
                            filename='weights-{epoch:02d}-{validation_loss:.2f}',
                            save_weights_only=True
                          )

    gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(max_epochs=training_config["n_epochs"],
                         callbacks=[checkpoint_callback],
                         gpus=gpus)
    return trainer


class LitTdoaCrnn(pl.LightningModule):
    """Pytorch lightning class used to abstract the
        training/validation/testing procedure of the TdoaCrnn neural network
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.target_key = self.config["model"]["target"]
        if self.config["model"]["selected_model"] == "crnn10":
            self.model = Crnn10()
        
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
        self.log("loss", avg_loss, on_step=False)
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("validation_loss", avg_loss, on_step=False)
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("test_loss", avg_loss, on_step=False)

    def configure_optimizers(self):
        lr = self.config["training"]["learning_rate"]
        optimizer = self.config["training"]["optimizer"]

        if optimizer == "sgd":
            return torch.optim.SGD(self.parameters(), lr=lr)
        elif optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=lr)
