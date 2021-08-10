from catalyst.runners import SupervisedRunner
import torch
import torch.optim as optim

from neural_tdoa.utils.callbacks import make_callbacks


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
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

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
