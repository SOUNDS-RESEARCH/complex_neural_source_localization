import librosa
import numpy as np
import torch

from complex_neural_source_localization.model import DOACNet


def test_crnn10():
    input_file_name = "tests/fixtures/0.0_split1_ir0_ov1_3.wav"

    signal = librosa.load(input_file_name, sr=24000,
                             mono=False, dtype=np.float32)[0][np.newaxis]

    signal = torch.Tensor(signal)
    model = DOACNet()

    result = model(signal)
