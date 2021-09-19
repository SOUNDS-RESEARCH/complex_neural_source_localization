import torch
from torch.nn import Module, MSELoss

from datasets.math_utils import compute_distance, gcc_phat, normalize_tdoa


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


def compute_tdoa_with_gcc_phat(x1, x2, fs, mic_positions):
    mic_distance = compute_distance(mic_positions[0], mic_positions[1], mode="torch")
    cc, lag_indexes = gcc_phat(x1, x2, fs)
    tdoa = lag_indexes[torch.argmax(torch.abs(cc))]
    normalized_tdoa = normalize_tdoa(tdoa, mic_distance)

    clamped_tdoa = torch.clamp(normalized_tdoa, 0, 1)
    return clamped_tdoa