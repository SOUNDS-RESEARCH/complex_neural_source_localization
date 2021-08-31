import pytorch_lightning as pl
import shutil
import torch

from datasets.dataset import TdoaDataset
from datasets.math_utils import compute_distance, gcc_phat, normalize_tdoa
from neural_tdoa.metrics import Loss, average_rms_error
from neural_tdoa.model import TdoaCrnn10

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
        y = y["target"]
        predictions = self.model(x)

        loss = self.loss(predictions, y)
        rms = average_rms_error(y, predictions)

        self.log("train_loss", loss)
        self.log("train_rms", rms, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        X, Y = batch

        mic_coordinates = Y["mic_coordinates"]
        Y = Y["target"]
        predictions = self.model(X)

        loss = self.loss(predictions, Y)

        rms = average_rms_error(predictions, Y)
        self.log("validation_loss", loss, prog_bar=True)
        self.log("validation_rms", rms, prog_bar=True)

        validation_config = self.config["validation_dataset"]
        fs = validation_config["base_sampling_rate"]
        
        tdoas_gcc_phat = torch.zeros_like(Y)
        
        for i, x in enumerate(X):
            tdoas_gcc_phat[i] = _compute_tdoa_with_gcc_phat(
                                x[0],
                                x[1],
                                fs,
                                mic_coordinates[i]
                               )

        rms_gcc = average_rms_error(Y, tdoas_gcc_phat)
            
        self.log("validation_rms_gcc", rms_gcc, prog_bar=True)

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

    training_config = config["training"]
    batch_size = training_config["batch_size"]
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

    
    gpus = 1 if torch.cuda.is_available() else 0
    tb_logger = pl_loggers.TensorBoardLogger("logs/")

    trainer = pl.Trainer(max_epochs=training_config["num_epochs"],
                         log_every_n_steps=training_config["log_every_n_steps"],
                         gpus=gpus, logger=tb_logger)
    trainer.fit(model, dataset_train, val_dataloaders=dataset_val)

    if training_config["delete_datasets_after_training"]:
        shutil.rmtree(config["training_dataset"]["dataset_dir"])
        shutil.rmtree(config["validation_dataset"]["dataset_dir"])


def _compute_tdoa_with_gcc_phat(x1, x2, fs, mic_positions):
    mic_distances = compute_distance(mic_positions[0], mic_positions[1], mode="torch")
    cc, lag_indexes = gcc_phat(x1, x2, fs)
    tdoa = lag_indexes[torch.argmax(torch.abs(cc))]
    normalized_tdoa = normalize_tdoa(tdoa, mic_distances)

    return normalized_tdoa
