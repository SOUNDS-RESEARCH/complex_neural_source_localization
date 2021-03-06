import torch

from torch.nn import Module, CosineSimilarity, MSELoss, L1Loss


from torch.nn import Module


class AngularLoss(Module):
    def __init__(self, model_output_type="scalar"):
        # See https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html
        # for a related implementation used for NLP
        super().__init__()

        dim = 1 if model_output_type == "scalar" else 2
        self.cosine_similarity = CosineSimilarity(dim=dim)
        # dim=0 -> batch | dim=1 -> time steps | dim=2 -> azimuth

    def forward(self, model_output, targets, mean_reduce=True):
        values = 1 - self.cosine_similarity(model_output, targets)
        if mean_reduce:
            values = values.mean()
        return values


class CustomL1Loss(Module):
    def __init__(self):
        super().__init__()

        self.mse = L1Loss(reduction="none")
        self.target_key = "source_coordinates"

    def forward(self, model_output, targets, mean_reduce=True):
        targets = targets[self.target_key]
        if model_output.shape != targets.shape:
            raise ValueError(
                "Model output's shape is {}, target's is {}".format(
                    model_output.shape, targets.shape
            ))
        loss = self.mse(model_output, targets)

        if mean_reduce:
            loss = loss.mean()
        
        return loss


class PitLoss(Module):
    def __init__(self):
        """Permutation-Invariant Training loss

        Args:
            loss_type (string): angular | magnitude
            model_output_type (str, optional): "scalar" or. Defaults to "scalar".
            mean_reduce (bool, optional): Whether to give out the batch mean or not

        """

        super().__init__()
        
        # 2 sources, therefore 2 keys
        self.target_keys = ["azimuth_2d_point", "azimuth_2d_point_2"]
        self.loss = AngularLoss()

    def forward(self, model_output, targets, mean_reduce=True):
        target_0 = targets[self.target_keys[0]]
        target_1 = targets[self.target_keys[1]]

        model_output_0 = model_output[:, 0:2]
        model_output_1 = model_output[:, 2:4]

        if model_output[:, 0:2].shape != target_0.shape:
            raise ValueError(
                "Model output's shape is {}, target's is {}".format(
                    model_output.shape, target_0.shape
            ))

        # Compute loss for every permutation: indexes 0:2 represent the first source prediction,
        # While indexes [2:4] represent the second one
        loss_0 = self.loss(model_output_0, target_0, mean_reduce=False) + self.loss(model_output_1, target_1, mean_reduce=False)
        loss_1 = self.loss(model_output_1, target_0, mean_reduce=False) + self.loss(model_output_0, target_1, mean_reduce=False)

        loss = torch.stack([loss_0, loss_1], dim=1)
        loss = loss.min(dim=1)[0]
        
        if mean_reduce:
            loss = loss.mean()/2
        
        return loss


LOSS_NAME_TO_CLASS_MAP = {
    "angular": AngularLoss,
    "l1": CustomL1Loss,
    "angular_pit": PitLoss
}

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

# def _create_activity_masks(start, end, max_duration_in_seconds, size, cuda=True):
#     masks = torch.stack([
#         _create_activity_mask(s, e, max_duration_in_seconds, size)
#         for s, e in zip(start, end)
#     ])
#     masks = masks.unsqueeze(2)
#     masks = masks.tile((1, 1, 2)) # In order to multiply by 2d array        

#     if cuda:
#         masks = masks.cuda()
#     return masks


# def _create_activity_mask(start, end, max_duration_in_seconds, size):
#     activity_mask = torch.zeros(size, dtype=torch.bool)

#     time_step_duration_in_seconds = max_duration_in_seconds/size
#     start_cell = floor(start/time_step_duration_in_seconds)
#     end_cell = ceil(end/time_step_duration_in_seconds)

#     activity_mask[start_cell:end_cell] = True
    
#     return activity_mask


# class Loss(Module):
#     def __init__(self, loss_type="angular", model_output_type="scalar"):
#         """Class abstracting the loss function

#         Args:
#             loss_type (string): angular | magnitude
#             model_output_type (str, optional): "scalar" or. Defaults to "scalar".

#         """

#         super().__init__()
        
#         self.target_key = "azimuth_2d_point"

#         if loss_type == "angular":
#             self.loss = AngularLoss(model_output_type)
#         elif loss_type == "magnitude":
#             self.loss = MagnitudeLoss()
#         elif loss_type == "angular_and_magnitude":
#             self.loss = AngularAndMagnitudeLoss(model_output_type)

        
#         self.model_output_type = model_output_type

#     def forward(self, model_output, targets, mean_reduce=True):
#         # if self.model_output_type == "frame":
#         #     activity_mask_size = model_output.shape[1]
#         #     activity_masks = _create_activity_masks(
#         #                         targets["start_time"], targets["end_time"],
#         #                         self.frame_size_in_seconds, activity_mask_size,
#         #                         model_output.is_cuda)

#         #     targets = targets[self.target_key].unsqueeze(1)
            
#         #     targets = targets*activity_masks
#         #     model_output = model_output*activity_masks
#         # else:
#         targets = targets[self.target_key]

#         if model_output.shape != targets.shape:
#             raise ValueError(
#                 "Model output's shape is {}, target's is {}".format(
#                     model_output.shape, targets.shape
#             ))
        
#         return self.loss(model_output, targets, mean_reduce=mean_reduce)

