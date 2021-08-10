import torch
import torch.nn as nn

from complexPyTorch.complexLayers import ComplexConv2d, ComplexBatchNorm2d
from complexPyTorch.complexFunctions import complex_relu
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d

from neural_tdoa.utils.initializers import init_gru, init_linear_layer
from neural_tdoa.feature_extractors import (
    MfccArray, StftArray, StftMagnitudeArray
)
from neural_tdoa.utils.show import show_params, show_model


class TdoaCrnn10(nn.Module):
    def __init__(self, model_config, dataset_config):

        super().__init__()

        self.n_model_output = 1 # The regressed normalized TDOA from 0-1
        self.model_config = model_config
        n_input_channels = len(dataset_config["mic_coordinates"])
        self._create_feature_extractor_layer(model_config, dataset_config)
        self._create_conv_layers(n_input_channels,
                                 model_config["n_conv_layers"],
                                 model_config["n_output_channels"])
        self._create_gru_layer(model_config["n_output_channels"])
        self._create_output_layer(model_config["n_output_channels"])

        show_model(self)
        show_params(self)

    def _create_conv_layers(self, n_input_channels, n_layers, max_filters):
        n_layer_outputs = [
            max_filters//(2**(n_layers - i))
            for i in range(1, n_layers + 1)
        ]
        
        if self.are_features_complex:
            n_layer_outputs = [n//2 for n in n_layer_outputs]
    
        self.n_conv_blocks = len(n_layer_outputs)
        
        n_layer_inputs = [n_input_channels] + n_layer_outputs
        self.conv_blocks = nn.ModuleList([
            ConvBlock(
                n_layer_inputs[i],
                n_layer_inputs[i+1],
                self.are_features_complex
            )
            for i in range(self.n_conv_blocks)
        ])
        
    def _create_feature_extractor_layer(self, model_config, dataset_config):
        feature_type = model_config["feature_type"]

        if feature_type == "stft":
            self.feature_extractor = StftArray(model_config)
            self.are_features_complex = True
        elif feature_type == "stft_magnitude":
            self.feature_extractor = StftMagnitudeArray(model_config)
            self.are_features_complex = False
        elif feature_type == "mfcc":
            self.feature_extractor = MfccArray(model_config, dataset_config)
            self.are_features_complex = False

    def _create_gru_layer(self, n_output_channels):
        self.gru = nn.GRU(
            input_size=n_output_channels, hidden_size=n_output_channels//2,
            num_layers=1, batch_first=True, bidirectional=True
        )
        init_gru(self.gru)

    def _create_output_layer(self, n_input_channels):
        self.fc_output = nn.Linear(n_input_channels,
                                   self.n_model_output,
                                   bias=True)
        init_linear_layer(self.fc_output)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor):  Raw multichannel audio tensor of shape 
                            (batch_size, n_channels, time_steps)
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, n_model_output)
        """        

        x = self.feature_extractor(x)
        # Feature extractor output: (batch_size, n_channels, time_steps, freq_bins)
        for i in range(self.n_conv_blocks):
            x = self.conv_blocks[i](x)
        # Conv layer output: (batch_size, feature_maps, freq_bins, time_steps)

        if self.are_features_complex:
            x = _convert_complex_tensor_to_real(x)
        # Output: (batch_size, 2*feature_maps, freq_bins, time_steps)

        x = torch.mean(x, dim=2)
        # Aggregated conv output (batch_size, feature_maps, time_steps)

        # GRU input: (batch_size, time_steps, feature_maps):"""
        x = x.transpose(1, 2)
        (x, _) = self.gru(x)
        x = self.fc_output(x)
        # Output: (batch_size, time_steps, n_output)"""

        x = torch.mean(x, dim=1)
        # Output: (batch_size, 1)

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
