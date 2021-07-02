import torchaudio

from neural_tdoa.models.dccrn.conv_stft import MultichannelConvSTFT


def test_multichannel_conv_stft():
    sample_path = "tests/fixtures/fold1_room1_mix001.wav"
    signal = torchaudio.load(sample_path)[0].unsqueeze(0)

    win_len = 400
    win_inc = 100
    fft_len = 512
    win_type = "hanning"
    feature_type = 'complex'
    
    spec_array = MultichannelConvSTFT(win_len, win_inc, fft_len, win_type, feature_type)
    result = spec_array(signal)

    assert result.shape == (1, 8, 257, 14403)
    """Batch size, num_channels*2, num_freq_channels, time_steps"""
