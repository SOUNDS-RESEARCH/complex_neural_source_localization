import torch
import torch.nn as nn

from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d

from tdoa.math_utils import denormalize, normalize

from neural_tdoa.utils.initializers import init_gru, init_linear_layer
from neural_tdoa.feature_extractors import (
    DecoupledStftArray, MagnitudeStftArray, StftArray
)
from neural_tdoa.utils.load_config import load_config


class TdoaCrnn(nn.Module):
    def __init__(self, model_config=None, max_tdoa=1/343):
        if model_config is None:
            model_config = load_config("model")

        super().__init__()

        self.n_model_output = 1 # The regressed normalized TDOA from 0-1
        self.n_input_channels = 2 # Two microphones
        self.max_tdoa = max_tdoa # Used for normalizing/denormalizing the network's output

        self.model_config = model_config
        self.pool_type = model_config["pool_type"]
        self._create_feature_extractor_layer(model_config)
        self._create_conv_layers(self.n_input_channels, model_config)
        self._create_gru_layer(model_config["n_output_channels"])
        self._create_output_layer(model_config["n_output_channels"])

    def _create_conv_layers(self, n_input_channels, model_config):
        n_layers = model_config["n_conv_layers"]
        max_filters = model_config["n_output_channels"]
        conv_type = model_config["conv_type"]
        pool_type = model_config["pool_type"]
        pool_size = model_config["pool_size"]
        
        n_layer_outputs = [
            max_filters//(2**(n_layers - i))
            for i in range(1, n_layers + 1)
        ]
    
        self.n_conv_blocks = len(n_layer_outputs)
        
        n_layer_inputs = [n_input_channels] + n_layer_outputs
        self.conv_blocks = nn.ModuleList([
            ConvBlock(
                n_layer_inputs[i],
                n_layer_inputs[i+1],
                conv_type,
                pool_type,
                list(pool_size)
            )
            for i in range(self.n_conv_blocks)
        ])
        
    def _create_feature_extractor_layer(self, model_config):
        feature_type = model_config["feature_type"]

        if feature_type == "stft_magnitude":
            self.feature_extractor = MagnitudeStftArray(model_config)
        elif feature_type == "decoupled_stft":
            self.feature_extractor = DecoupledStftArray(model_config)
        elif feature_type == "stft":
            self.feature_extractor = StftArray(model_config)

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

    def forward(self, x, normalized=True):
        """
        Args:
            x (torch.Tensor):  Raw multichannel audio tensor of shape 
                            (batch_size, n_channels, time_steps)
            normalized (bool): If set to true, return is between [0,1].
                                Else, return is between [-self.min_tdoa, self.max_tdoa]
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, n_model_output)
        """
        x = self.feature_extractor(x)

        # Feature extractor output: (batch_size, n_channels, time_steps, freq_bins)
        for i in range(self.n_conv_blocks):
            x = self.conv_blocks[i](x)
        # Conv layer output: (batch_size, feature_maps, freq_bins, time_steps)

        x = self._aggregate_features(x, 2)
        # Aggregated conv output (batch_size, feature_maps, time_steps)

        # GRU input: (batch_size, time_steps, feature_maps):"""
        x = x.transpose(1, 2)
        (x, _) = self.gru(x)
        x = self.fc_output(x)
        # Output: (batch_size, time_steps, 1)"""

        x = self._aggregate_features(x, 1)
        # Output: (batch_size, 1)
 
        x = torch.sigmoid(x)

        if not normalized:
            x = denormalize(x, -self.max_tdoa, self.max_tdoa)
        
        return x

    def _aggregate_features(self, x, dim):
        if self.pool_type == "avg":
            return torch.mean(x, dim=dim)
        elif self.pool_type == "max":
            return torch.max(x, dim=dim)


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, conv_type, pool_type, pool_size, kernel_size=3):
        super().__init__()

        self.conv_type = conv_type
        self.pool_type = pool_type
        self.pool_size = pool_size

        if conv_type == "depthwise_separable":
            self.depth_conv = Conv2d(in_channels=input_dim,
                                out_channels=input_dim,
                                kernel_size=kernel_size, groups=input_dim,
                                padding="same")
            self.point_conv = Conv2d(in_channels=input_dim,
                                out_channels=output_dim,
                                kernel_size=1,)
            self.conv_block = nn.Sequential(self.depth_conv, self.point_conv)
        elif conv_type == "conv2d":
            self.conv_block = Conv2d(input_dim, output_dim, kernel_size)

        self.bn_block = BatchNorm2d(output_dim)
        
        if pool_type == "avg":
            self.pool_block = nn.AvgPool2d(pool_size)
        elif pool_type == "max":
            self.pool_block = nn.MaxPool2d(pool_size)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.bn_block(x)
        x = torch.relu(x)

        x = self.pool_block(x)

        return x
