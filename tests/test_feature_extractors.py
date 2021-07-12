import numpy as np
import torchaudio

from neural_tdoa.feature_extractors import (
    StftArray, MfccArray
)


def test_mel_spectrogram_array():
    sample_path = "tests/fixtures/fold1_room1_mix001.wav"
    signal = torchaudio.load(sample_path)[0].unsqueeze(0)
    mel_spec_array = MfccArray()

    result = mel_spec_array(signal)

    assert result.shape == (1, 4, 64, 3001)
    """Batch size, n_array, n_mels, time_steps"""


def test_spectrogram_array():
    sample_path = "tests/fixtures/fold1_room1_mix001.wav"
    signal = torchaudio.load(sample_path)[0].unsqueeze(0)
    spec_array = StftArray()

    result = spec_array(signal)

    assert result.shape == (1, 4, 512, 3001)
    """Batch size, n_array, n_fft//2 + 1, time_steps"""
