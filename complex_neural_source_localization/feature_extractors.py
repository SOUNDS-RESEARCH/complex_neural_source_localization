import torch

from torch.nn import Module
from torchaudio.transforms import MelSpectrogram


class MfccArray(Module):
    def __init__(self, model_config, dataset_config):

        super().__init__()

        self.mel_spectrogram = MelSpectrogram(
            sample_rate=dataset_config["sr"],
            n_fft=model_config["n_fft"],
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
        self.onesided = model_config["use_onesided_fft"]

    def forward(self, X):
        "Expected input has shape (batch_size, n_arrays, time_steps)"

        result = []
        n_arrays = X.shape[1]

        for i in range(n_arrays):
            x = X[:, i, :]
            stft_output = torch.stft(x, self.n_fft, onesided=self.onesided, return_complex=True)
            result.append(
                stft_output[:, 1:, :]
            ) # Ignore frequency 0
        
        result = torch.stack(result, dim=1)
        return result


class MagnitudeStftArray(StftArray):
    def forward(self, X):
        stft = super().forward(X)
        return stft.abs()


class StftPhaseArray(StftArray):
    def forward(self, X):
        stft = super().forward(X)
        return stft.angle()


class DecoupledStftArray(StftArray):
    "Stft where the real and imaginary channels are modeled as separate channels"
    def forward(self, X):

        stft = super().forward(X)

        # stft.real.shape = (batch_size, num_mics, num_channels, time_steps)
        result = torch.cat((stft.real, stft.imag), dim=2)   
        
        return result


class CrossSpectra(Module):
    def __init__(self, model_config):

        super().__init__()

        self.n_fft = model_config["n_fft"]
        self.stft_extractor = StftArray(model_config)

    def forward(self, X):
        "Expected input has shape (batch_size, n_channels, time_steps)"
        batch_size, n_channels, time_steps = X.shape

        stfts = self.stft_extractor(X)
        # (batch_size, n_channels, n_freq_bins, n_time_bins)
        cross_spectra = []

        for sample_idx in range(batch_size):
            # TODO: maybe there is a way to vectorize the loops below,
            # although it would probably repeat many operations
            sample_cross_spectra = []
            for channel_1 in range(n_channels):
                for channel_2 in range(channel_1, n_channels):
                    cross_spectrum = stfts[sample_idx][channel_1]*stfts[sample_idx][channel_2].conj()
                    sample_cross_spectra.append(cross_spectrum)
            cross_spectra.append(
                torch.stack(sample_cross_spectra, axis=0)
            )
        
        result = torch.stack(cross_spectra, dim=0)
        return result
