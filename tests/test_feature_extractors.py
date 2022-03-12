from numpy import unwrap
import torchaudio

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from complex_neural_source_localization.feature_extractors import (
    StftArray, CrossSpectra
)
from complex_neural_source_localization.utils.model_visualization import (
    plot_multichannel_spectrogram
)


def test_spectrogram_array():
    cfg = _load_config()

    sample_path = "tests/fixtures/fold1_room1_mix001.wav"
    signal = torchaudio.load(sample_path)[0].unsqueeze(0)

    spec_array = StftArray(cfg["model"])

    result = spec_array(signal)

    plot_multichannel_spectrogram(result[0], output_path="tests/temp/multichannel_stft.png")

    assert result.shape == (1, 4, 512, 5626)
    """Batch size, n_array, n_fft//2 + 1, time_steps"""


def test_cross_spectrum_array():
    cfg = _load_config()

    sample_path = "tests/fixtures/fold1_room1_mix001.wav"
    signal, sr = torchaudio.load(sample_path)
    cross_spec_array = CrossSpectra(cfg["model"])

    result = cross_spec_array(signal.unsqueeze(0))

    plot_multichannel_spectrogram(result[0], output_path="tests/temp/cross_spectra.png", mode="row", unwrap=True, db=True)

    assert result.shape == (1, 10, 512, 5626)
    """Batch size, n_array, n_fft//2 + 1, time_steps"""



def _load_config():
    GlobalHydra.instance().clear()

    initialize(config_path="../config", job_name="test_app")
    cfg = compose(config_name="config")

    return cfg
