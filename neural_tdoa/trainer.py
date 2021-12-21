import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import ModelCheckpoint

from neural_tdoa.metrics import Loss, average_rms_error, compute_tdoa_with_gcc_phat
from neural_tdoa.model import TdoaCrnn


def create_trainer(training_config):
    checkpoint_callback = ModelCheckpoint(
                            monitor="validation_rms",
                            save_last=True,
                            filename='weights-{epoch:02d}-{validation_rms:.2f}',
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

        self.model = TdoaCrnn(config["model"])
        self.loss = Loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y[self.target_key]
        predictions = self.model(x)

        loss = self.loss(predictions, y)
        rms = average_rms_error(y, predictions, max_tdoa=self.model.max_tdoa)

        output_dict = {
            "loss": loss,
            "rms": rms
        }

        return output_dict
    
    def validation_step(self, batch, batch_idx):
        X, Y = batch

        mic_coordinates = Y["mic_coordinates"]
        Y = Y[self.target_key]
        predictions = self.model(X)

        loss = self.loss(predictions, Y)

        rms = average_rms_error(predictions, Y, max_tdoa=self.model.max_tdoa)

        validation_config = self.config["validation_dataset"]
        fs = validation_config["base_sampling_rate"]
        
        tdoas_gcc_phat = torch.zeros_like(Y)
        for i, x in enumerate(X):
            tdoas_gcc_phat[i] = compute_tdoa_with_gcc_phat(
                                    x[0],
                                    x[1],
                                    fs,
                                    mic_coordinates[i]
                                )

        rms_gcc = average_rms_error(Y, tdoas_gcc_phat, max_tdoa=self.model.max_tdoa)
            
        output_dict = {
            "loss": loss,
            "rms": rms,
            "rms_gcc": rms_gcc
        }
        
        return output_dict

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_rms = torch.stack([x['rms'] for x in outputs]).mean()
        
        self.log("loss", avg_loss, on_step=False)
        self.log("rms", avg_rms, on_step=False)
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_rms = torch.stack([x['rms'] for x in outputs]).mean()
        avg_rms_gcc = torch.stack([x['rms_gcc'] for x in outputs]).mean()

        self.log("validation_loss", avg_loss, on_step=False)
        self.log("validation_rms", avg_rms, on_step=False)
        self.log("validation_rms_gcc", avg_rms_gcc, on_step=False)
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_rms = torch.stack([x['rms'] for x in outputs]).mean()
        avg_rms_gcc = torch.stack([x['rms_gcc'] for x in outputs]).mean()

        self.log("test_loss", avg_loss, on_step=False)
        self.log("test_rms", avg_rms, on_step=False)
        self.log("test_rms_gcc", avg_rms_gcc, on_step=False)
        

    def configure_optimizers(self):
        lr = self.config["training"]["learning_rate"]
        optimizer = self.config["training"]["optimizer"]

        if optimizer == "sgd":
            return torch.optim.SGD(self.parameters(), lr=lr)
        elif optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=lr)
