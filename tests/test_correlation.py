from matplotlib import pyplot as plt
import librosa
import os
from pyroomasync.room import ConnectedShoeBox
from pyroomasync.simulator import simulate
from tdoa.tdoa import compute_correlations
from tdoa.visualization import plot_correlations


def test_plot_correlation():
    _test_plot_correlation("correlation")

def test_plot_gcc_phat():
    _test_plot_correlation("gcc-phat")


def _test_plot_correlation(correlation_mode):
    os.makedirs("tests/temp", exist_ok=True)
    temp_file_path = f"tests/temp/tdoa_plot_correlation_{correlation_mode}.png"
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    input_signal, fs = librosa.load("tests/fixtures/p225_001.wav")

    room_dim = [5, 5, 3]
    source_location = [2, 3, 1]
    mic_locations = [[2, 2, 1], [3, 3, 1], [4, 4, 1]]

    room = ConnectedShoeBox(room_dim, fs=fs)
    room.add_source(source_location, input_signal)
    room.add_microphone_array(mic_locations)

    simulation_results = simulate(room)
    correlations = compute_correlations(simulation_results, fs, correlation_mode)

    # plot_correlations(correlations, output_path=temp_file_path)

    # plt.close()
    # assert os.path.exists(temp_file_path)
