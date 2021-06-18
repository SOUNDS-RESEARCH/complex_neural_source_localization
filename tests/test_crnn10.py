import numpy as np
import torchaudio

from yoho.model.common.crnn10 import Crnn10


def test_forward():
    sample_path = "tests/fixtures/fold1_room1_mix001.wav"
    signal = torchaudio.load(sample_path)[0][np.newaxis]
    n_model_output = 20
    model = Crnn10(n_model_output)
    result = model.forward(signal)
    assert result.shape == (1, 375, 20)
