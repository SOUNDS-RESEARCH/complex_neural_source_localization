import numpy as np

from pyroomasync.room import ConnectedShoeBox
from pyroomasync.simulator import simulate
from tdoa.tdoa import compute_tdoas


def test_compute_tdoas():
    def sinusoid(freq_in_hz, duration, sr):
        linear_samples = np.arange(duration*sr)
        return np.sin(linear_samples*freq_in_hz)
    
    fs = 48000
    input_signal = sinusoid(10, 1, fs)

    room_dim = [4, 6, 3]
    source_location = [1, 5, 1]
    mic_locations = [[2, 2, 1], [3, 3, 1]]

    room = ConnectedShoeBox(room_dim)
    room.add_source(source_location, input_signal)
    room.add_microphone_array(mic_locations)

    simulation_results = simulate(room)
    tdoas = compute_tdoas(simulation_results, room.base_fs)
    # Currently no assertions, just seeing if it goes through
