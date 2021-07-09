import torch
import torch.nn as nn

from complexPyTorch.complexLayers import ComplexConv2d, ComplexBatchNorm2d
from complexPyTorch.complexFunctions import complex_relu
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d

from neural_tdoa.models.common.model_utilities import init_gru, init_layer
from neural_tdoa.models.common.feature_extractors import MfccArray, StftArray
from neural_tdoa.models.common.show import show_params, show_model

from neural_tdoa.models.settings import OUTPUT_CHANNELS

from datasets.settings import N_MICS


class TdoaBaseCrnn10(nn.Module):
    def __init__(
        self,
        feature_extractor="stft",
        n_input_channels=N_MICS,
        output_channels=OUTPUT_CHANNELS
    ):

        super().__init__()

        self.n_model_output = 1 # The regressed normalized TDOA from 0-1

        if feature_extractor == "stft":
            self.feature_extractor = StftArray()
            self.are_features_complex = True
        elif feature_extractor == "mfcc":
            self.feature_extractor = MfccArray()
            self.are_features_complex = False

        self._create_conv_layers(n_input_channels, output_channels)

        self.gru = nn.GRU(
            input_size=output_channels, hidden_size=output_channels//2,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self.fc_output = nn.Linear(output_channels, self.n_model_output, bias=True)

        self.init_weights()

        show_model(self)
        show_params(self)

    def _create_conv_layers(self, n_input_channels, max_filters):
        if self.are_features_complex:
            n_layer_inputs = [
                n_input_channels,
                max_filters//16,
                max_filters//8,
                max_filters//4,
                max_filters//2
            ]
        else:
            n_layer_inputs = [
                n_input_channels,
                max_filters//8,
                max_filters//4,
                max_filters//2,
                max_filters
            ]

        self.n_conv_blocks = len(n_layer_inputs) - 1

        self.conv_blocks = nn.ModuleList([
            ConvBlock(
                n_layer_inputs[i],
                n_layer_inputs[i+1],
                self.are_features_complex
            )
            for i in range(self.n_conv_blocks)
        ])

    def init_weights(self):
        init_gru(self.gru)
        init_layer(self.fc_output)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor):  Raw multichannel audio tensor of shape 
                            (batch_size, n_channels, time_steps)
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, n_model_output)
        """        

        x = self.feature_extractor(x)
        # feature_extractor_output: (batch_size, n_channels, time_steps, freq_bins)
        for i in range(self.n_conv_blocks):
            x = self.conv_blocks[i](x)
        # Conv output: (batch_size, feature_maps, freq_bins, time_steps)

        if self.are_features_complex:
            x = _convert_complex_tensor_to_real(x)

        x = torch.mean(x, dim=2)
        # Aggregated conv output (batch_size, feature_maps, time_steps)

        # GRU input: (batch_size, time_steps, feature_maps):"""
        x = x.transpose(1, 2)
        (x, _) = self.gru(x)
        x = self.fc_output(x)
        # Output: (batch_size, time_steps, n_output)"""

        # Experimental: Output 2d (batch_size, 1)
        # (xmax, _) = torch.max(x, dim=1)
        x = torch.mean(x, dim=1)
        # x = x + xmax

        return torch.sigmoid(x)
    
    def _aggregate_features(self, x):
        if self.pool_type == "avg":
            return torch.mean(x, dim=2)


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, is_complex, kernel_size=3):
        super().__init__()

        conv_module = ComplexConv2d if is_complex else Conv2d
        self.conv_block = conv_module(input_dim, output_dim, kernel_size) 

        bn_module = ComplexBatchNorm2d if is_complex else BatchNorm2d
        self.bn_block = bn_module(output_dim)

        self.activation = complex_relu if is_complex else torch.relu 

    def forward(self, x):
        x = self.conv_block(x)
        x = self.bn_block(x)
        x = self.activation(x)

        return x


def _convert_complex_tensor_to_real(complex_tensor):
    """Convert a complex tensor of shape
    (batch_size, num_channels, time_steps, freq_bins) 
    into a real tensor of shame
    (batch_size, 2*num_channels, time_steps, freq_bins)"""

    real_tensor = complex_tensor.real
    imag_tensor = complex_tensor.imag

    result_tensor = torch.cat([real_tensor, imag_tensor], dim=1)

    return result_tensor
