import torch
import torch.nn as nn

from neural_tdoa.models.common.model_utilities import SpectrogramArray
from neural_tdoa.models.common.complex import (
    ComplexLinear, ComplexConv2d, complex_relu
)

from neural_tdoa.models.settings import (
    N_FFT, HOP_LENGTH
)
from datasets.settings import N_MICS

CNN14_OUTPUT_FEATURE_SIZE = 2048
TDOA_CLASSES_NUM = 1


class TdoaSimpleCnn(nn.Module):
    def __init__(
            self, n_fft=N_FFT, hop_length=HOP_LENGTH):

        super().__init__()

        self.spectrogram = SpectrogramArray(
            n_fft=n_fft,
            hop_length=hop_length
        )

        self.conv_block1 = ComplexConv2d(N_MICS, 64)
        self.conv_block2 = ComplexConv2d(64, 128)
        self.fc1 = ComplexLinear(128, 1, bias=True)

    def forward(self, input):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram(input)

        x = torch.stack((x.real, x.imag), dim=-1)

        x = complex_relu(self.conv_block1(x))

        x = complex_relu(self.conv_block2(x))

        x = torch.mean(x, dim=3)
        x = torch.mean(x, dim=2)
        x = self.fc1(x)
        x = torch.abs(x)
        x = torch.sigmoid(x)

        return x


