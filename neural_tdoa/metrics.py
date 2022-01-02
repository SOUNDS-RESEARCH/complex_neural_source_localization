import torch
from torch.nn import Module, MSELoss, CosineSimilarity

from tdoa.math_utils import (
    compute_distance, gcc_phat, normalize_tdoa, denormalize
)

# Loss functions

class CartesianLoss(Module):
    def __init__(self):
        super().__init__()

        self.mse = MSELoss()
        self.complex_mse = ComplexMSELoss()

    def forward(self, model_output, targets):
        if model_output.shape != targets.shape:
            raise ValueError(
                "Model output's shape is {}, target's is {}".format(
                    model_output.shape, targets.shape
            ))
        
        if model_output.dtype == torch.complex64:
            return self.complex_mse(model_output, targets)
        else:
            return self.mse(model_output, targets)


class AngularLoss(Module):
    def __init__(self):
        super().__init__()
        self.cosine_similarity = CosineSimilarity()

    def forward(self, model_output, targets):
        values = 1 - self.cosine_similarity(model_output, targets)
        return values.mean()


class ComplexMSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, model_output, targets):
        return complex_mse_loss(model_output, targets)


def complex_mse_loss(model_output, targets):
    if model_output.shape != targets.shape:
        raise ValueError(
            "Model output's shape is {}, target's is {}".format(
                model_output.shape, targets.shape
        ))
    
    error = model_output - targets
    mean_squared_error = (error*error.conj()).sum()/error.shape[0]
    mean_squared_error = mean_squared_error.real # Imaginary part is 0
    return mean_squared_error


# Metrics

def average_rms_error(y_true, y_pred, max_tdoa=None):
    if max_tdoa is not None:
        min_tdoa = -max_tdoa
        y_true = denormalize(y_true, min_tdoa, max_tdoa)
        y_pred = denormalize(y_pred, min_tdoa, max_tdoa)

    with torch.no_grad():
        return torch.sqrt(complex_mse_loss(y_pred, y_true))


def compute_tdoa_with_gcc_phat(x1, x2, fs, mic_positions):
    mic_distance = compute_distance(mic_positions[0], mic_positions[1], mode="torch")
    cc, lag_indexes = gcc_phat(x1, x2, fs)
    tdoa = lag_indexes[torch.argmax(torch.abs(cc))]
    normalized_tdoa = normalize_tdoa(tdoa, mic_distance)

    clamped_tdoa = torch.clamp(normalized_tdoa, 0, 1)
    return clamped_tdoa
