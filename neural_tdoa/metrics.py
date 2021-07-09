import torch


def average_rms_error(y_true, y_pred):
    with torch.no_grad():
        return torch.mean(torch.sqrt((y_true - y_pred)**2))
