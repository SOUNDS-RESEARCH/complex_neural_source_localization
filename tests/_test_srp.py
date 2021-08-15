import librosa
from matplotlib import pyplot as plt
import numpy as np

from pyroomasync.room import ConnectedShoeBox
from pyroomasync.simulator import simulate
from tdoa.srp import srp_phat


def test_srp_phat():
    fs = 48000
    input_signal, _ = librosa.load("tests/fixtures/p225_001.wav", fs)

    room_dim = [4, 6, 3]
    source_location = [1, 5, 1]
    mic_locations = [[1, 1, 1], [2, 2, 1], [3, 3, 1]]

    room = ConnectedShoeBox(room_dim)
    room.add_source(source_location, input_signal)
    room.add_microphone_array(mic_locations)

    simulation_results = simulate(room)
    srp = srp_phat(simulation_results, room)
    azimuth_recon = srp.azimuth_recon
    alpha_recon = srp.grid.values[srp.src_idx]

    plt.scatter([np.cos(azimuth_recon)], [np.sin(azimuth_recon)])
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    srp.polar_plt_dirac(save_fig=True, file_name="tests/temp/doa_srp.pdf")
    # Currently no assertions, just seeing if it goes through
