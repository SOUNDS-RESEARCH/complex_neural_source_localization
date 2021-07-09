import torch
import torch.nn as nn

from complexPyTorch.complexLayers import ComplexConv2d, ComplexBatchNorm2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d

from neural_tdoa.models.common.model_utilities import init_gru, init_layer
from neural_tdoa.models.common.feature_extractors import StftArray

from neural_tdoa.models.settings import (
    POOL_SIZE, POOL_TYPE, MAX_FILTERS
)
from neural_tdoa.models.dccrn.show import show_params, show_model

from datasets.settings import SR
from datasets.settings import N_MICS


class TdoaBaseCrnn10(nn.Module):
    def __init__(
        self,
        feature_extractor="stft",
        n_input_channels=N_MICS,
        sr=SR,
        pool_type=POOL_TYPE, pool_size=POOL_SIZE, max_filters=MAX_FILTERS
    ):

        super().__init__()

        self.n_model_output = 1 # TDOA
        self.pool_type = pool_type
        self.pool_size = pool_size

        if feature_extractor == "stft":
            self.feature_extractor = StftArray()
        else:
            raise ValueError("Only 'stft' feature extractor is currently implemented")

        is_complex = True if feature_extractor == "stft" else False
        self._create_conv_layers(n_input_channels, max_filters, is_complex)

        self.gru = nn.GRU(
            input_size=max_filters, hidden_size=max_filters//2,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self.fc_output = nn.Linear(max_filters, self.n_model_output, bias=True)

        self.init_weights()

        show_model(self)
        show_params(self)

    def _create_conv_layers(self, n_input_channels, max_filters, is_complex):
        n_layer_inputs = [
            n_input_channels*2,
            max_filters//8,
            max_filters//4,
            max_filters//2,
            max_filters
        ]
        self.n_conv_blocks = len(n_layer_inputs) - 1

        self.conv_blocks = nn.ModuleList([
            _conv_block(n_layer_inputs[i], n_layer_inputs[i+1], is_complex)
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
        #x = x.transpose(2, 3)
        # feature_extractor_output: (batch_size, n_channels, mel_bins, time_steps)
        # VERIFY! ^
        for i in range(self.n_conv_blocks):
            x = self.conv_blocks[i](x)
        # Conv output: (batch_size, feature_maps, time_steps, mel_bins)

        x = self._aggregate_features(x)
        # Aggregated conv output (batch_size, feature_maps, time_steps)

        x = x.transpose(1, 2)
        # GRU input: (batch_size, time_steps, feature_maps):"""
        (x, _) = self.gru(x)
        x = self.fc_output(x)
        # Output: (batch_size, time_steps, n_output)"""

        # Experimental: Output 2d (batch_size, 1)
        (x1, _) = torch.max(x, dim=1)
        x2 = torch.mean(x, dim=1)
        x = x1 + x2

        return torch.sigmoid(x)
    
    def _aggregate_features(self, x):
        if self.pool_type == "avg":
            return torch.mean(x, dim=2)
        else:
            return torch.max(x, dim=2)[0]


def _conv_block(input_dim, output_dim, is_complex):
    conv_module = ComplexConv2d if is_complex else Conv2d
    bn_module = ComplexBatchNorm2d if is_complex else BatchNorm2d

    block = nn.Sequential(
        conv_module(
            input_dim,
            output_dim
        ),
        bn_module(output_dim),
        nn.PReLU()
    )

    return block