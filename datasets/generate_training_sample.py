import os

from pyroomasync import ConnectedShoeBox, simulate

from datasets.logger import save_signals
from datasets.math_utils import compute_distance, compute_tdoa, normalize_tdoa
from datasets.generate_random_configs import (
    generate_random_microphone_coordinates,
    generate_random_source_coordinates,
    generate_random_source_signal,
    generate_random_delay
)


def generate_training_sample(training_sample_config, log_melspectrogram=False):
    output_signals = _simulate(training_sample_config)

    os.makedirs(training_sample_config["signals_dir"], exist_ok=True)
    save_signals(output_signals,
                 training_sample_config["sr"],
                 training_sample_config["signals_dir"],
                 log_melspectrogram)


def _simulate(sample_config):
    base_sr = sample_config["sr"]
    source_signal = sample_config["source_signal"]
    num_input_samples = source_signal.shape[0]
    mic_delays = sample_config["mic_delays"]
    # Convert delay to Milliseconds
    mic_delays = [delay/1000 for delay in mic_delays]

    room = ConnectedShoeBox(sample_config["room_dims"], fs=base_sr)

    room.add_microphone_array(sample_config["mic_coordinates"],
                              delay=mic_delays)

    room.add_source(sample_config["source_coordinates"], source_signal)

    signals = simulate(room)

    signals = _trim_recorded_signals(signals, num_input_samples, mic_delays, base_sr)
    
    return signals


def generate_random_training_sample_config(base_config):
    mic_coordinates = base_config["mic_coordinates"]
    room_dims = base_config["room_dims"]

    if mic_coordinates is None:
        mic_coordinates = generate_random_microphone_coordinates(
                                                room_dims)

    source_coordinates = generate_random_source_coordinates(
                                            room_dims,
                                            mic_coordinates[0][2])

    tdoa, normalized_tdoa = _compute_tdoa(source_coordinates, mic_coordinates)

    mic_delays = [
        base_config["mic_0_delay"],
        generate_random_delay(*base_config["mic_1_delay_range"])
    ]
    mic_sampling_rates = [
        base_config["mic_0_sampling_rate"],
        generate_random_delay(*base_config["mic_1_sampling_rate_range"])
    ]
    
    source_signal, gain = generate_random_source_signal(
                            base_config["base_sampling_rate"],
                            base_config["sample_duration_in_secs"],
                            mic_delays)

    return {
        "room_dims": room_dims,
        "source_x": source_coordinates[0],
        "source_y": source_coordinates[1],
        "source_coordinates": source_coordinates,
        "mic_coordinates": mic_coordinates,
        "mic_delays": mic_delays,
        "mic_sampling_rates": mic_sampling_rates,
        "tdoa": tdoa,
        "normalized_tdoa": normalized_tdoa,
        "sr": base_config["base_sampling_rate"],
        "source_signal": source_signal,
        "gain": gain
    }


def _compute_tdoa(source_coordinates, mic_coordinates):
    mic_distance = compute_distance(mic_coordinates[0], mic_coordinates[1])
    tdoa = compute_tdoa(source_coordinates, mic_coordinates)
    normalized_tdoa = normalize_tdoa(tdoa, mic_distance)

    return tdoa, normalized_tdoa


def _trim_recorded_signals(signals, num_output_samples, mic_delays, sr):
    max_delay = max(mic_delays)
    max_delay_in_samples = int(max_delay*sr)

    # Remove silence in the beginning of signals which might make
    # Detecting the delays "too easy", then truncate to input size
    signals = signals[:, max_delay_in_samples:]
    signals = signals[:, :num_output_samples]

    return signals
