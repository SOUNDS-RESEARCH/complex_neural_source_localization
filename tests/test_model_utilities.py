import numpy as np
import torchaudio

from neural_tdoa.models.common.model_utilities import (
    SpectrogramArray, MelSpectrogramArray
)


def test_mel_spectrogram_array():
    sample_path = "tests/fixtures/fold1_room1_mix001.wav"
    signal = torchaudio.load(sample_path)[0].unsqueeze(0)
    mel_spec_array = MelSpectrogramArray()

    result = mel_spec_array(signal)

    assert result.shape == (1, 4, 64, 3001)
    """Batch size, n_array, n_mels, time_steps"""


def test_spectrogram_array():
    sample_path = "tests/fixtures/fold1_room1_mix001.wav"
    signal = torchaudio.load(sample_path)[0].unsqueeze(0)
    spec_array = SpectrogramArray()

    result = spec_array(signal)

    assert result.shape == (1, 4, 513, 3001)
    """Batch size, n_array, n_fft//2 + 1, time_steps"""
