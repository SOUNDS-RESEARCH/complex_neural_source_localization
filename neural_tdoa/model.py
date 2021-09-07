import torch
import torch.nn as nn

from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d

from neural_tdoa.utils.initializers import init_gru, init_linear_layer
from neural_tdoa.feature_extractors import (
    MfccArray, RealStftArray, StftMagnitudeArray
)
from neural_tdoa.utils.load_config import load_config


class TdoaCrnn10(nn.Module):
    def __init__(self, model_config=None, dataset_config=None):
        if model_config is None:
            model_config = load_config("model")
        if dataset_config is None:
            dataset_config = load_config("training_dataset")

        super().__init__()

        self.n_model_output = 1 # The regressed normalized TDOA from 0-1
        self.model_config = model_config
        n_input_channels = dataset_config["n_mics"]
        self._create_feature_extractor_layer(model_config, dataset_config)
        self._create_conv_layers(n_input_channels,
                                 model_config["n_conv_layers"],
                                 model_config["n_output_channels"],
                                 model_config["conv_type"])
        self._create_gru_layer(model_config["n_output_channels"])
        self._create_output_layer(model_config["n_output_channels"])

    def _create_conv_layers(self, n_input_channels, n_layers, max_filters, conv_type):
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
                conv_type
            )
            for i in range(self.n_conv_blocks)
        ])
        
    def _create_feature_extractor_layer(self, model_config, dataset_config):
        feature_type = model_config["feature_type"]

        if feature_type == "stft_magnitude":
            self.feature_extractor = StftMagnitudeArray(model_config)
        elif feature_type == "stft":
            self.feature_extractor = RealStftArray(model_config)
        elif feature_type == "mfcc":
            self.feature_extractor = MfccArray(model_config, dataset_config)

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
    def __init__(self, input_dim, output_dim, conv_type, kernel_size=3):
        super().__init__()

        self.conv_type = conv_type

        if conv_type == "depthwise_separable":
            depth_conv = Conv2d(in_channels=input_dim,
                                    out_channels=input_dim,
                                    kernel_size=kernel_size, groups=input_dim)
            point_conv = Conv2d(in_channels=input_dim,
                                    out_channels=output_dim,
                                    kernel_size=1)
            self.conv_block = nn.Sequential(depth_conv, point_conv)
        elif conv_type == "conv2d":
            self.conv_block = Conv2d(input_dim, output_dim, kernel_size)

        self.bn_block = BatchNorm2d(output_dim)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.bn_block(x)
        x = torch.relu(x)

        return x

