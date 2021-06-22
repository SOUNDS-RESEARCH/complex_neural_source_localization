import torch
import torch.nn as nn

from neural_tdoa.models.common.model_utilities import (
    ConvBlock, init_gru, init_layer, MelSpectrogramArray)
from neural_tdoa.models.settings import (
    N_FFT, N_MELS, HOP_LENGTH, POOL_SIZE, POOL_TYPE, MAX_FILTERS
)
from datasets.settings import SR
from datasets.settings import N_MICS


class TdoaCrnn10(nn.Module):
    def __init__(
        self,
        n_model_input_in_seconds=1,
        n_input_channels=N_MICS,
        sr=SR, n_fft=N_FFT, n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        pool_type=POOL_TYPE, pool_size=POOL_SIZE,
        max_filters=MAX_FILTERS
    ):

        super().__init__()

        self.n_model_input = n_model_input_in_seconds*sr
        self.n_model_output = 1 # TDOA
        self.pool_type = pool_type
        self.pool_size = pool_size

        self.mel_spectrogram = MelSpectrogramArray(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        self._create_conv_layers(n_input_channels, max_filters)

        self.gru = nn.GRU(
            input_size=max_filters, hidden_size=max_filters//2,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self.fc_output = nn.Linear(max_filters, self.n_model_output, bias=True)

        self.init_weights()

    def _create_conv_layers(self, n_input_channels, max_filters):
        n_layer_outputs = [max_filters//8, max_filters//4, max_filters//2, max_filters]

        self.conv_block1 = ConvBlock(
            in_channels=n_input_channels, out_channels=n_layer_outputs[0])
        self.conv_block2 = ConvBlock(
            in_channels=n_layer_outputs[0], out_channels=n_layer_outputs[1])
        self.conv_block3 = ConvBlock(
            in_channels=n_layer_outputs[1], out_channels=n_layer_outputs[2])
        self.conv_block4 = ConvBlock(
            in_channels=n_layer_outputs[2], out_channels=n_layer_outputs[3])

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

        x = self.mel_spectrogram(x) 
        x = x.transpose(2, 3)
        # feature_extractor_output: (batch_size, n_channels, time_steps, mel_bins)"
        x = self.conv_block1(x)
        x = self.conv_block2(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block3(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block4(x, self.pool_type, pool_size=self.pool_size)
        # Conv output: (batch_size, feature_maps, mel_bins, time_steps)

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
            return torch.mean(x, dim=3)
        else:
            return torch.max(x, dim=3)[0]
