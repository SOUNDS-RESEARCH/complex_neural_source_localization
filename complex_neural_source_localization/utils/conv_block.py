import torch.nn as nn

from complex_neural_source_localization.utils.complexPyTorch.complexLayers import (
    ComplexAmpTanh, ComplexConv2d, ComplexBatchNorm2d, ComplexDropout,
    ComplexReLU, ComplexTanh, ComplexPReLU, ComplexAvgPool2d
)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                kernel_size=(3,3), stride=(1,1),
                padding=(1,1), pool_size=(2, 2),
                block_type="real_double",
                init=False,
                dropout_rate=0.1,
                activation="relu"):
        
        super().__init__()
        self.block_type = block_type
        self.pool_size=pool_size
        self.dropout_rate = dropout_rate

        if "complex" in block_type:
            conv_block = ComplexConv2d
            bn_block = ComplexBatchNorm2d
            dropout_block = ComplexDropout
            if activation == "relu":
                self.activation = ComplexReLU()
            elif activation == "amp_tanh":
                self.activation = ComplexAmpTanh()
            elif activation == "tanh":
                self.activation = ComplexTanh()
            elif activation == "prelu":
                self.activation = ComplexPReLU()

            self.pooling = ComplexAvgPool2d(pool_size)
            self.is_real = False
            out_channels = out_channels//2
        else:
            conv_block = nn.Conv2d
            bn_block = nn.BatchNorm2d
            dropout_block = nn.Dropout
            self.activation = nn.ReLU()
            self.pooling = nn.AvgPool2d(pool_size)
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

        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        if "double" in self.block_type:
            x = self.activation(self.bn2(self.conv2(x)))
        x = self.pooling(x)
        
        if self.dropout_rate > 0:
            x = self.dropout(x)
        return x
