# Credits to Yin Cao et al:
# https://github.com/yinkalario/Two-Stage-Polyphonic-Sound-Event-Detection-and-Localization/blob/master/models/model_utilities.py


import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from complex_neural_source_localization.utils.complexPyTorch.complexLayers import (
    ComplexConv2d, ComplexBatchNorm2d, ComplexDropout
)
from complex_neural_source_localization.utils.complexPyTorch.complexFunctions import (
    complex_avg_pool2d, complex_relu
)

def interpolate(x, ratio):
    """
    Interpolate the x to have equal time steps as targets
    Input:
        x: (batch_size, time_steps, class_num)
    Output:
        out: (batch_size, time_steps*ratio, class_num) 
    """

    x = x.transpose(1, 2) # Transpose as interpolate works on last axis
    x = F.interpolate(x, scale_factor=ratio)
    x = x.transpose(1, 2) # Transpose back
    
    return x


def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a convolutional or linear layer"""
    classname = layer.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.constant_(layer.bias, 0.0)


def init_gru(rnn):
    """Initialize a GRU layer. """
    
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                kernel_size=(3,3), stride=(1,1),
                padding=(1,1), pool_size=(2, 2),
                block_type="real_double",
                init=False,
                dropout_rate=0.1):
        
        super().__init__()
        self.block_type = block_type
        self.pool_size=pool_size
        self.dropout_rate = dropout_rate

        if "complex" in block_type:
            conv_block = ComplexConv2d
            bn_block = ComplexBatchNorm2d
            dropout_block = ComplexDropout
            self.activation = complex_relu
            self.pooling = complex_avg_pool2d
            self.is_real = False
            out_channels = out_channels//2
        else:
            conv_block = nn.Conv2d
            bn_block = nn.BatchNorm2d
            dropout_block = nn.Dropout
            self.activation = F.relu
            self.pooling = F.avg_pool2d
            self.is_real = True

        self.conv1 = conv_block(in_channels=in_channels, 
                            out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride,
                            padding=padding, bias=False)
        self.bn1 = bn_block(out_channels)
        self.dropout = dropout_block(dropout_rate)

        if "double" in block_type: 
            self.conv2 = conv_block(in_channels=out_channels, 
                                out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=False)           
            self.bn2 = bn_block(out_channels)
        
        if "real" in block_type and init: # Complex initialization not yet supported
            self.init_weights()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.bn1)
        if self.block_type == "real_double":
            init_layer(self.conv2)
            init_layer(self.bn2)
        
    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        if "double" in self.block_type:
            x = self.activation(self.bn2(self.conv2(x)))
        x = self.pooling(x, kernel_size=self.pool_size)
        
        if self.dropout_rate > 0:
            x = self.dropout(x)
        return x
