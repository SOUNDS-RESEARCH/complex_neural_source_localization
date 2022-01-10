import torch

from math import ceil, floor
from torch.nn import Module, MSELoss, CosineSimilarity

from tdoa.math_utils import denormalize

# Loss functions

class Loss(Module):
    def __init__(self, loss_type, model_output_type="scalar"):
        """Class abstracting the loss function

        Args:
            loss_type (string): angular | magnitude
            model_output_type (str, optional): "scalar" or. Defaults to "scalar".

        """

        super().__init__()
        
        self.target_key = "azimuth_2d_point"

        if loss_type == "angular":
            self.loss = AngularLoss(model_output_type)
        elif loss_type == "magnitude":
            self.loss = MagnitudeLoss()
        elif loss_type == "angular_and_magnitude":
            self.loss = AngularAndMagnitudeLoss(model_output_type)

        
        self.model_output_type = model_output_type

    def forward(self, model_output, targets):
        activity_mask_size = model_output.shape[1]
        if self.model_output_type == "frame":
            activity_masks = _create_activity_masks(
                                targets["start_time"], targets["end_time"],
                                self.frame_size_in_seconds, activity_mask_size,
                                model_output.is_cuda)

            targets = targets[self.target_key].unsqueeze(1)
            
            targets = targets*activity_masks
            model_output = model_output*activity_masks
        else:
            targets = targets[self.target_key]

        if model_output.shape != targets.shape:
            raise ValueError(
                "Model output's shape is {}, target's is {}".format(
                    model_output.shape, targets.shape
            ))
        
        return self.loss(model_output, targets)


class AngularLoss(Module):
    def __init__(self, model_output_type="scalar"):
        # See https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html
        # for a related implementation used for NLP
        super().__init__()

        dim = 1 if model_output_type == "scalar" else 2
        self.cosine_similarity = CosineSimilarity(dim=dim)
        # dim=0 -> batch | dim=1 -> time steps | dim=2 -> azimuth

    def forward(self, model_output, targets):
        values = 1 - self.cosine_similarity(model_output, targets)
        return values.mean()


class MagnitudeLoss(Module):
    def __init__(self):
        super().__init__()
    def forward(self, model_output, targets):
        model_output_norm = torch.norm(model_output, dim=1)
        targets_norm = torch.norm(targets, dim=1)
        return torch.mean((model_output_norm - targets_norm)**2)


class AngularAndMagnitudeLoss(Module):
    def __init__(self):
        super().__init__()

        self.angular_loss = AngularLoss()
        self.magnitude_loss = MagnitudeLoss()

    def forward(self, model_output, targets):
        angular_loss = self.angular_loss(model_output, targets)
        magnitude_loss = self.magnitude_loss(model_output, targets)
        return angular_loss + magnitude_loss


# class ComplexMSELoss(Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, model_output, targets):
#         return complex_mse_loss(model_output, targets)


# def complex_mse_loss(model_output, targets):
#     if model_output.shape != targets.shape:
#         raise ValueError(
#             "Model output's shape is {}, target's is {}".format(
#                 model_output.shape, targets.shape
#         ))
    
#     error = model_output - targets
#     mean_squared_error = (error*error.conj()).sum()/error.shape[0]
#     mean_squared_error = mean_squared_error.real # Imaginary part is 0
#     return mean_squared_error



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
