from datasets.dataset import TdoaDataset
from neural_tdoa.metrics import Loss, average_rms_error
from neural_tdoa.model import TdoaCrnn10
import pytorch_lightning as pl
import torch

from pytorch_lightning import loggers as pl_loggers


class LitTdoaCrnn10(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = TdoaCrnn10(config["model"], config["training_dataset"])
        self.loss = Loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.model(x)

        loss = self.loss(x, y)
        rms = average_rms_error(y, x)

        self.log("train_loss", loss)
        self.log("train_rms", rms, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.model(x)

        loss = self.loss(x, y)

        rms = average_rms_error(y, x)
        self.log("validation_loss", loss, prog_bar=True)
        self.log("validation_rms", rms, prog_bar=True)
        
    def configure_optimizers(self):
        lr = self.config["training"]["learning_rate"]
        optimizer = self.config["training"]["optimizer"]

        if optimizer == "sgd":
            return torch.optim.SGD(self.parameters(), lr=lr)
        elif optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=lr)


def train_pl(config):
    model = LitTdoaCrnn10(config)
    
    dataset_train = TdoaDataset(config["training_dataset"])
    dataset_val = TdoaDataset(config["validation_dataset"])

    batch_size = config["training"]["batch_size"]
    dataset_train = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                pin_memory=True,
                                                drop_last=False,
                                                num_workers=2)
    dataset_val = torch.utils.data.DataLoader(dataset_val,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              drop_last=False,
                                              num_workers=2)

    num_epochs = config["training"]["num_epochs"]
    
    gpus = 1 if torch.cuda.is_available() else 0
    tb_logger = pl_loggers.TensorBoardLogger("logs/")

    trainer = pl.Trainer(max_epochs=num_epochs, log_every_n_steps=10,
                         gpus=gpus, logger=tb_logger)
    trainer.fit(model, dataset_train, val_dataloaders=dataset_val)
