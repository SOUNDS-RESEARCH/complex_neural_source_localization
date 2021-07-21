import os

from pyroomasync import ConnectedShoeBox, simulate

from datasets.logger import save_signals
from datasets.math_utils import compute_tdoa, compute_tdoa_range, normalize
from datasets.generate_random_configs import (
    generate_random_microphone_coordinates,
    generate_random_source_coordinates,
    generate_random_source_signal
)


def generate_training_sample(training_sample_config, log_melspectrogram=False):
    output_signals = _simulate(training_sample_config)

    os.makedirs(training_sample_config["signals_dir"], exist_ok=True)
    save_signals(output_signals,
                 training_sample_config["sr"],
                 training_sample_config["signals_dir"],
                 log_melspectrogram)


def _simulate(training_sample_config):
    base_sr = training_sample_config["sr"]
    source_signal = training_sample_config["source_signal"]
    num_input_samples = source_signal.shape[0]
    mic_delays = training_sample_config["mic_delays"]

    room = ConnectedShoeBox(training_sample_config["room_dims"], fs=base_sr)

    room.add_microphone_array(training_sample_config["mic_coordinates"],
                              delay=mic_delays)

    room.add_source(training_sample_config["source_coordinates"], source_signal)

    signals = simulate(room)

    signals = _trim_recorded_signals(signals, num_input_samples, mic_delays, base_sr)
    
    return signals


def generate_random_training_sample_config(base_config):
    mic_coordinates = base_config["mic_coordinates"]
    if mic_coordinates is None:
        mic_coordinates = generate_random_microphone_coordinates(
                                                base_config["room_dims"])

    source_coordinates = generate_random_source_coordinates(
                                            base_config["room_dims"],
                                            mic_coordinates[0][2])

    tdoa, normalized_tdoa = _compute_tdoa(source_coordinates, mic_coordinates)

    source_signal, gain = generate_random_source_signal(base_config)
    return {
        "room_dims": base_config["room_dims"],
        "source_x": source_coordinates[0],
        "source_y": source_coordinates[1],
        "source_coordinates": source_coordinates,
        "mic_coordinates": mic_coordinates,
        "mic_delays": base_config["mic_delays"],
        "tdoa": tdoa,
        "normalized_tdoa": normalized_tdoa,
        "sr": base_config["base_sampling_rate"],
        "source_signal": source_signal,
        "gain": gain
    }


def _compute_tdoa(source_coordinates, mic_coordinates):
    min_tdoa, max_tdoa = compute_tdoa_range(mic_coordinates)
    tdoa = compute_tdoa(source_coordinates, mic_coordinates)
    normalized_tdoa = normalize(tdoa, min_tdoa, max_tdoa)

    return tdoa, normalized_tdoa


def _trim_recorded_signals(signals, num_output_samples, mic_delays, sr):
    max_delay = max(mic_delays)
    max_delay_in_samples = int(max_delay*sr)

    # Remove silence in the beginning of signals which might make
    # Detecting the delays "too easy", then truncate to input size
    signals = signals[:, max_delay_in_samples:]
    signals = signals[:, :num_output_samples]

    return signals
