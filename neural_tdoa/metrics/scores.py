import torch


def average_l1_distance(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))
