from complex_neural_source_localization.dataset import load_multichannel_wav
from complex_neural_source_localization.model import DOACNet


def test_tdoa_crnn10_with_stft():
    _test_tdoa_crnn10("stft")


def test_tdoa_crnn10_with_mfcc():
    _test_tdoa_crnn10("mfcc")


def _test_tdoa_crnn10(feature_type):

    model = DOACNet(n_sources=1)
    
    sample = load_multichannel_wav("tests/fixtures/0.0_split1_ir0_ov1_3.wav", 16000, 1)

    model_output = model(sample.unsqueeze(0))

    assert model_output.shape == (1, 2)
