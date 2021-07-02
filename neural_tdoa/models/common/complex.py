import torch.nn as nn
import torch


def complex_relu(x):
    re = torch.relu(x[..., 0])
    #im = torch.relu(x[..., 1])

    return torch.stack((re, re), dim=-1)


class ComplexConv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super().__init__()
        self.real_layer = nn.Conv2d(in_features, out_features, kernel_size, stride, padding,
                 dilation, groups, bias)
        self.imaginary_layer = nn.Conv2d(in_features, out_features, kernel_size, stride, padding,
                 dilation, groups, bias)
    def forward(self, x):
        return _apply_complex(x, self.real_layer, self.imaginary_layer)


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.real_layer = nn.Linear(in_features, out_features, bias)
        self.imaginary_layer = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        return _apply_complex(x, self.real_layer, self.imaginary_layer)


def _apply_complex(x, real_layer, imaginary_layer):
    
    re = real_layer(x[..., 0]) # - imaginary_layer(x[..., 1])
    #im = real_layer(x[..., 1]) + imaginary_layer(x[..., 0])

    return torch.stack((re, re), dim=-1)
