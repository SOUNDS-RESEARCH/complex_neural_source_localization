import torch
from torch.nn import Module, MSELoss


class Loss(Module):
    def __init__(self):
        super().__init__()

        self.mse = MSELoss()

    def forward(self, model_output, targets):
        if model_output.shape != targets.shape:
            raise ValueError(
                "Model output's shape is {}, target's is {}".format(
                    model_output.shape, targets.shape
            ))
        return self.mse(model_output, targets)


def average_rms_error(y_true, y_pred):
    with torch.no_grad():
        return torch.mean(torch.sqrt((y_true - y_pred)**2))
