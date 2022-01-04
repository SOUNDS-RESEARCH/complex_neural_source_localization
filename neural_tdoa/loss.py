import torch

from math import ceil, floor
from torch.nn import Module, MSELoss, CosineSimilarity

from tdoa.math_utils import denormalize

# Loss functions

class Loss(Module):
    def __init__(self, loss_type, frame_size_in_seconds=1):
        super().__init__()

        if loss_type == "real_cartesian":
            self.loss = MSELoss()
            self.target_key = "azimuth_2d_point"
        elif loss_type == "complex_cartesian":
            self.loss = ComplexMSELoss()
            self.target_key = "azimuth_complex_point"
        elif loss_type == "real_angular":
            self.loss = AngularLoss()
            self.target_key = "azimuth_2d_point"
        
        self.frame_size_in_seconds = frame_size_in_seconds

    def forward(self, model_output, targets):
        activity_mask_size = model_output.shape[1]
        activity_masks = _create_activity_masks(
                            targets["start_time"], targets["end_time"],
                            self.frame_size_in_seconds, activity_mask_size,
                            model_output.is_cuda)

        targets = targets[self.target_key].unsqueeze(1)
        
        targets = targets*activity_masks
        masked_model_output = model_output*activity_masks

        if masked_model_output.shape != targets.shape:
            raise ValueError(
                "Model output's shape is {}, target's is {}".format(
                    model_output.shape, targets.shape
            ))
        
        return self.loss(masked_model_output, targets)


class AngularLoss(Module):
    def __init__(self):
        # See https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html
        # for a related implementation used for NLP
        super().__init__()
        self.cosine_similarity = CosineSimilarity(dim=2)
        # dim=0 -> batch | dim=1 -> time steps | dim=2 -> azimuth

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


def _create_activity_masks(start, end, max_duration_in_seconds, size, cuda=True):
    masks = torch.stack([
        _create_activity_mask(s, e, max_duration_in_seconds, size)
        for s, e in zip(start, end)
    ])
    masks = masks.unsqueeze(2)
    masks = masks.tile((1, 1, 2)) # In order to multiply by 2d array        

    if cuda:
        masks = masks.cuda()
    return masks


def _create_activity_mask(start, end, max_duration_in_seconds, size):
    activity_mask = torch.zeros(size, dtype=torch.bool)

    time_step_duration_in_seconds = max_duration_in_seconds/size
    start_cell = floor(start/time_step_duration_in_seconds)
    end_cell = ceil(end/time_step_duration_in_seconds)

    activity_mask[start_cell:end_cell] = True
    
    return activity_mask
