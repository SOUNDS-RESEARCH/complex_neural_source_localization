from matplotlib import pyplot as plt
import librosa
import os
import pyroomacoustics as pra

from pyroomasync.settings import SPEED_OF_SOUND
from pyroomasync.room import ConnectedShoeBox
from pyroomasync.simulator import simulate
from tdoa.math import compute_distance
from tdoa.tdoa import compute_tdoas
from tdoa.visualization import plot_location_candidates, plot_top_candidate_points


def test_plot_location_candidates_with_correlation():
    _test_plot_location_candidates("correlation")


def test_plot_location_candidates_with_gcc_phat():
    _test_plot_location_candidates("gcc-phat")


def test_plot_location_candidates_with_gcc_phat_wall_absorption():
    _test_plot_location_candidates("gcc-phat", max_absorption=True)


def test_plot_location_candidates_with_correlation_wall_absorption():
    _test_plot_location_candidates("correlation", max_absorption=True)


def test_plot_top_candidate_points_correlation():
    _test_plot_top_candidate_points("correlation")


def test_plot_top_candidate_points_correlation():
    _test_plot_top_candidate_points("gcc-phat")


def _test_plot_top_candidate_points(mode):
    plt.close()
    os.makedirs("tests/temp", exist_ok=True)
    temp_file_path = f"tests/temp/lowest_error_candidates_{mode}.png"
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    room, simulation_results = _simulate()
    tdoas = compute_tdoas(simulation_results, room.fs, mode)
    distances = {
        key:SPEED_OF_SOUND*tdoa for key, tdoa in tdoas.items()
    }
    plot_top_candidate_points(room, distances, temp_file_path)


def test_plot_top_candidate_gt():
    plt.close()
    os.makedirs("tests/temp", exist_ok=True)
    temp_file_path = "tests/temp/lowest_error_candidates_gt.png"
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    room, _ = _simulate()
    
    mic_locations = [m.loc for m in room.microphones.mic_array]
    source_location = room.sources.source_array[0].loc

    dist0 = compute_distance(mic_locations[0], source_location)
    dist1 = compute_distance(mic_locations[1], source_location)
    dist2 = compute_distance(mic_locations[2], source_location)

    # Using theoretical tdoas
    distances = {
        (0, 1): abs(dist0 - dist1),
        (0, 2): abs(dist0 - dist2), 
        (1, 2): abs(dist1 - dist2)
    }

    plot_top_candidate_points(room, distances, temp_file_path)


def test_plot_two_mics_one_source_with_ground_truth_tdoas():
    plt.close()
    os.makedirs("tests/temp", exist_ok=True)
    temp_file_path = "tests/temp/tdoa_distance_2_1_heatmap_gt.png"
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    room, _ = _simulate_2_1()
    mic_locations = [m.loc for m in room.microphones.mic_array]
    source_location = room.sources.source_array[0].loc

    dist0 = compute_distance(mic_locations[0], source_location)
    dist1 = compute_distance(mic_locations[1], source_location)

    # Using theoretical tdoas
    doas = {
        (0, 1): abs(dist0 - dist1)
    }
    plot_location_candidates(room, doas, temp_file_path)

    assert os.path.exists(temp_file_path)


def test_plot_location_candidates_with_ground_truth_tdoas():
    plt.close()
    os.makedirs("tests/temp", exist_ok=True)
    temp_file_path = "tests/temp/tdoa_distance_heatmap_gt.png"
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    room, _ = _simulate()
    mic_locations = [m.loc for m in room.microphones.mic_array]
    source_location = room.sources.source_array[0].loc

    dist0 = compute_distance(mic_locations[0], source_location)
    dist1 = compute_distance(mic_locations[1], source_location)
    dist2 = compute_distance(mic_locations[2], source_location)

    # Using theoretical tdoas
    doas = {
        (0, 1): abs(dist0 - dist1),
        (0, 2): abs(dist0 - dist2), 
        (1, 2): abs(dist1 - dist2)
    }
    plot_location_candidates(room, doas, temp_file_path)

    assert os.path.exists(temp_file_path)


def _test_plot_location_candidates(correlation_mode, max_absorption=False):
    plt.close()
    os.makedirs("tests/temp", exist_ok=True)
    temp_file_path = f"tests/temp/tdoa_distance_heatmap_{correlation_mode}_max_absorption={max_absorption}.png"
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    room, simulation_results = _simulate(max_absorption)

    tdoas = compute_tdoas(simulation_results, room.fs, correlation_mode)
    doas = {
        key:SPEED_OF_SOUND*tdoa for key, tdoa in tdoas.items()
    }
    plot_location_candidates(room, doas, temp_file_path)

    assert os.path.exists(temp_file_path)


def _simulate(max_absorption=False):
    input_signal, fs = librosa.load("tests/fixtures/p225_001.wav")

    room_dim = [5, 5, 3]
    source_location = [2, 3, 1]
    mic_locations = [[2, 2, 1], [3, 3, 1], [4, 4, 1]]

    if max_absorption:
        room = ConnectedShoeBox(room_dim, fs=fs, materials=pra.Material(1.0))
    else:
        room = ConnectedShoeBox(room_dim, fs=fs)
    room.add_source(source_location, input_signal)
    room.add_microphone_array(mic_locations)

    simulation_results = simulate(room)

    return room, simulation_results


def _simulate_2_1(max_absorption=False):
    input_signal, fs = librosa.load("tests/fixtures/p225_001.wav")

    room_dim = [5, 5, 3]
    source_location = [2, 3, 1]
    mic_locations = [[2, 2, 1], [3, 3, 1]]

    if max_absorption:
        room = ConnectedShoeBox(room_dim, fs=fs, materials=pra.Material(1.0))
    else:
        room = ConnectedShoeBox(room_dim, fs=fs)
    room.add_source(source_location, input_signal)
    room.add_microphone_array(mic_locations)

    simulation_results = simulate(room)

    return room, simulation_results