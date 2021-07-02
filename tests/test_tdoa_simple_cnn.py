import torchaudio

from neural_tdoa.models.tdoa_simple_cnn import TdoaSimpleCnn


def test_tdoa_simple_cnn():
    sample_path = "tests/fixtures/fold1_room1_mix001.wav"
    signal = torchaudio.load(sample_path)[0].unsqueeze(0)
    signal = signal[:, 0:2, :] # only get first 2 channels

    signal.shape
    model = TdoaSimpleCnn()

    result = model(signal)

