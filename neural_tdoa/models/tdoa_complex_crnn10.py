import torch
import torch.nn as nn

from neural_tdoa.models.common.model_utilities import (
    ConvBlock, init_gru, init_layer)
from neural_tdoa.models.settings import (
    N_FFT, N_MELS, HOP_LENGTH, POOL_SIZE, POOL_TYPE, MAX_FILTERS
)
from neural_tdoa.models.dccrn.conv_stft import MultichannelConvSTFT
from neural_tdoa.models.dccrn.complexnn import ComplexConv2d, ComplexBatchNorm

from datasets.settings import SR
from datasets.settings import N_MICS


class TdoaComplexCrnn10(nn.Module):
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

        win_len=400
        win_inc=100
        self.stft = MultichannelConvSTFT(win_len, win_inc, N_FFT, 'hanning', 'complex')

        self._create_conv_layers(n_input_channels, max_filters)

        self.gru = nn.GRU(
            input_size=max_filters, hidden_size=max_filters//2,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self.fc_output = nn.Linear(max_filters, self.n_model_output, bias=True)

        self.init_weights()

    def _create_conv_layers(self, n_input_channels, max_filters):
        n_layer_outputs = [max_filters//8, max_filters//4, max_filters//2, max_filters]

        self.conv_block1 = _conv_block(n_input_channels*2, n_layer_outputs[0]*2)
        self.conv_block2 = _conv_block(n_layer_outputs[0]*2, n_layer_outputs[1]*2)
        self.conv_block3 = _conv_block(n_layer_outputs[1]*2, n_layer_outputs[2]*2)
        self.conv_block4 = _conv_block(n_layer_outputs[2]*2, n_layer_outputs[3]*2)

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

        x = self.stft(x) 
        x = x[:, :, 1:]
        #x = x.transpose(2, 3)
        # feature_extractor_output: (batch_size, n_channels, mel_bins, time_steps)
        # VERIFY! ^
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
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


def _conv_block(input_dim, output_dim, kernel_size=5, use_cbn=True):
    block = nn.Sequential(
        #nn.ConstantPad2d([0, 0, 0, 0], 0),
        ComplexConv2d(
            input_dim,
            output_dim,
            kernel_size=(kernel_size, 2),
            stride=(2, 1),
            padding=(2, 1)
        ),
        nn.BatchNorm2d(
            output_dim) if not use_cbn else ComplexBatchNorm(output_dim),
        nn.PReLU()
    )

    return block