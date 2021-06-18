import numpy as np
import torchaudio

from yoho.model.common.multichannel_cnn_14 import MultichannelCnn14


# This works, but takes a lot of time. Uncomment for testing
# def test_multichannel_cnn14():
#     sample_path = "tests/fixtures/fold1_room1_mix001.wav"
#     signal = torchaudio.load(sample_path)[0][np.newaxis]
#     model = MultichannelCnn14()
#     result = model.forward(signal)

#     assert result["embedding"].shape == (1, 4, 2048)
