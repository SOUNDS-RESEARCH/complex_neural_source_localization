import torchaudio

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from complex_neural_source_localization.feature_extractors import (
    StftArray, MfccArray
)


# def test_mel_spectrogram_array():
#     cfg = _load_config()
    
#     sample_path = "tests/fixtures/fold1_room1_mix001.wav"
#     signal = torchaudio.load(sample_path)[0].unsqueeze(0)
#     mel_spec_array = MfccArray(cfg["model"], cfg["dataset"])

#     result = mel_spec_array(signal)

#     assert result.shape == (1, 4, 64, 3001)
#     """Batch size, n_array, n_mels, time_steps"""


def test_spectrogram_array():
    cfg = _load_config()

    sample_path = "tests/fixtures/fold1_room1_mix001.wav"
    signal = torchaudio.load(sample_path)[0].unsqueeze(0)
    spec_array = StftArray(cfg["model"])

    result = spec_array(signal)

    assert result.shape == (1, 4, 512, 3001)
    """Batch size, n_array, n_fft//2 + 1, time_steps"""


def _load_config():
    GlobalHydra.instance().clear()

    initialize(config_path="../config", job_name="test_app")
    cfg = compose(config_name="config")

    return cfg
