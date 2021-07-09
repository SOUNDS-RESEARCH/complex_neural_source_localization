import torch

from torch.nn import Module
from torchaudio.transforms import MelSpectrogram

from datasets.settings import SR
from neural_tdoa.models.settings import (
    N_FFT, N_MELS, HOP_LENGTH
)


class MelSpectrogramArray(Module):
    def __init__(self, sample_rate=SR,
                 n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):

        super().__init__()

        self.mel_spectrogram = MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels
        )

    def forward(self, X):
        "Expected input has shape (batch_size, n_arrays, time_steps)"

        result = []

        n_arrays = X.shape[1]

        for i in range(n_arrays):
            x = X[:, i, :]
            result.append(self.mel_spectrogram(x))

        return torch.stack(result, dim=1)


class SpectrogramArray(Module):
    def __init__(self,
                 n_fft=N_FFT, hop_length=HOP_LENGTH):

        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, X):
        "Expected input has shape (batch_size, n_arrays, time_steps)"

        result = []

        n_arrays = X.shape[1]

        for i in range(n_arrays):
            x = X[:, i, :]
            result.append(
                torch.stft(x, self.n_fft, self.hop_length, return_complex=True)
            )

        return torch.stack(result, dim=1)
