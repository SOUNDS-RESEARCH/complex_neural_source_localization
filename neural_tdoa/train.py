from catalyst.runners import SupervisedRunner
import torch
import torch.optim as optim

BATCH_SIZE = 32
NUM_EPOCHS = 8
LEARNING_RATE = 0.0001


def train(model, loss_function, dataset_train, dataset_val,
          batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
          learning_rate=LEARNING_RATE, callbacks=[]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    runner.train(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optimizer=optimizer,
        num_epochs=num_epochs,
        verbose=True,
        logdir="logs/",
        callbacks=callbacks
    )
