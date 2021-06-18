import numpy as np
import torchaudio

from yoho.model.common.model_utilities import MelSpectrogramArray


def test_mel_spectrogram_array():
    sample_path = "tests/fixtures/fold1_room1_mix001.wav"
    signal = torchaudio.load(sample_path)[0].unsqueeze(0)
    mel_spec_array = MelSpectrogramArray()

    result = mel_spec_array(signal)

    assert result.shape == (1, 4, 64, 6001)
    """Batch size, n_array, n_mels, time_steps"""
