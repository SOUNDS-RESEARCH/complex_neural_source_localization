import torch

from torch.nn import Module
from torchaudio.transforms import MelSpectrogram


class MfccArray(Module):
    def __init__(self, model_config, dataset_config):

        super().__init__()

        self.mel_spectrogram = MelSpectrogram(
            sample_rate=dataset_config["base_sampling_rate"],
            n_fft=model_config["n_fft"],
            hop_length=model_config["hop_length"],
            n_mels=model_config["n_mels"]
        )

    def forward(self, X):
        "Expected input has shape (batch_size, n_arrays, time_steps)"

        result = []

        n_arrays = X.shape[1]

        for i in range(n_arrays):
            x = X[:, i, :]
            result.append(self.mel_spectrogram(x))

        return torch.stack(result, dim=1)


class StftArray(Module):
    def __init__(self, model_config):

        super().__init__()

        self.n_fft = model_config["n_fft"]
        self.hop_length = model_config["hop_length"]

    def forward(self, X):
        "Expected input has shape (batch_size, n_arrays, time_steps)"

        result = []
        n_arrays = X.shape[1]

        for i in range(n_arrays):
            x = X[:, i, :]
            stft_output = torch.stft(x, self.n_fft, self.hop_length, return_complex=True)
            result.append(
                stft_output[:, 1:, :]
            ) # Ignore frequency 0

        return torch.stack(result, dim=1)


class StftMagnitudeArray(StftArray):
    def forward(self, X):
        stft = super().forward(X)
        return stft.abs()


class StftPhaseArray(StftArray):
    def forward(self, X):
        stft = super().forward(X)
        return stft.angle()
