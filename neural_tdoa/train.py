from datasets.dataset import TdoaDataset
from neural_tdoa.metrics import Loss, average_rms_error
from neural_tdoa.model import TdoaCrnn10
import pytorch_lightning as pl
import torch
from catalyst.runners import SupervisedRunner

from neural_tdoa.utils.callbacks import make_callbacks


class LitTdoaCrnn10(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = TdoaCrnn10(config["model"], config["dataset"])
        self.loss = Loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.model(x)

        loss = self.loss(x, y)

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.model(x)

        loss = self.loss(x, y)

        rms = average_rms_error(y, x)
        self.log("validation_loss", loss)
        self.log("validation_rms", rms)
        

    
    def configure_optimizers(self):
        lr = self.config["training"]["learning_rate"]
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        return optimizer


def train_pl(config):
    model = LitTdoaCrnn10(config)
    
    dataset_config = config["dataset"]
    dataset_train = TdoaDataset(dataset_config)
    dataset_val = TdoaDataset(dataset_config, is_validation=True)

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
    trainer = pl.Trainer(max_epochs=num_epochs, log_every_n_steps=10,
                         gpus=gpus)
    trainer.fit(model, dataset_train, val_dataloaders=dataset_val)


def train(training_config,
          model, loss_function, dataset_train, dataset_val,
          callbacks=None, log_dir="logs/"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = training_config["batch_size"]
    learning_rate = training_config["learning_rate"]
    num_epochs = training_config["num_epochs"]

    loaders = {
        "train": torch.utils.data.DataLoader(dataset_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             drop_last=False),
        "valid": torch.utils.data.DataLoader(dataset_val,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False)
    }

    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Loss
    criterion = loss_function.to(device)

    runner = SupervisedRunner(
        input_key="signals",
        output_key="model_output",
        target_key="targets",
        loss_key="loss"
    )

    if callbacks is None:
        callbacks = make_callbacks(log_dir)

    runner.train(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optimizer=optimizer,
        num_epochs=num_epochs,
        verbose=True,
        logdir=log_dir,
        callbacks=callbacks
    )
