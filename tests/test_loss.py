import librosa
import numpy as np
import torch

from complex_neural_source_localization.loss import Loss
from complex_neural_source_localization.model import Crnn10


def test_neural_tdoa_loss():

    loss_fn = Loss("real_angular")
    model = Crnn10()

    sample_path = "tests/fixtures/0.0_split1_ir0_ov1_3.wav"

    sample = librosa.load(sample_path, sr=24000, mono=False, dtype=np.float32)[0]
    sample = torch.from_numpy(sample).unsqueeze(0)

    target = {
        "azimuth_2d_point": torch.Tensor([[0.0, 1.0]]),
        "start_time": torch.Tensor([0.3]),
        "end_time": torch.Tensor([0.6])
    }

    model_output = model(sample)
    
    _ = loss_fn(model_output, target)