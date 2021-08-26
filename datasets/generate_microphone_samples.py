"""
Generate simulated recordings from an environment
containing two microphones and a source
"""

import librosa
import os
import random

from pyroomasync import ConnectedShoeBox, simulate

from datasets.logger import save_signals
from datasets.math_utils import compute_distance, compute_tdoa, normalize_tdoa
from datasets.generate_random_configs import (
    generate_random_microphone_coordinates,
    generate_random_source_coordinates,
    generate_random_source_signal,
    generate_random_delay
)


def generate_microphone_samples(config, log_melspectrogram=False):
    output_signals = _simulate(config)

    os.makedirs(config["signals_dir"], exist_ok=True)
    save_signals(output_signals,
                 config["sr"],
                 config["signals_dir"],
                 log_melspectrogram)


def _simulate(sample_config):
    base_sr = sample_config["sr"]
    source_signal = sample_config["source_signal"]
    num_input_samples = source_signal.shape[0]
    mic_delays = sample_config["mic_delays"]
    mic_gains = sample_config["mic_gains"]
    trim_beginning = sample_config["trim_beginning"]
    # Convert delay to Milliseconds
    mic_delays = [delay for delay in mic_delays]

    room = ConnectedShoeBox(sample_config["room_dims"], fs=base_sr)

    room.add_microphone_array(sample_config["mic_coordinates"],
                              delay=mic_delays,
                              gain=mic_gains)

    room.add_source(sample_config["source_coordinates"], source_signal)
    signals = simulate(room)

    signals = _trim_recorded_signals(signals,
                                     num_input_samples,
                                     mic_delays,
                                     base_sr,
                                     trim_beginning)
    
    return signals


def generate_random_training_sample_config(base_config):
    mic_coordinates = base_config["mic_coordinates"]
    room_dims = base_config["room_dims"]
    trim_beginning = base_config["trim_beginning"]

    if not mic_coordinates:
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
    mic_gains = [
        base_config["mic_0_gain"],
        base_config["mic_1_gain"]
    ]
    
    if "anechoic_samples" not in base_config:
        # Make random signal's duration bigger to trim silent beginning
        max_delay = max(mic_delays)
        total_duration = base_config["sample_duration_in_secs"] + max_delay

        source_signal, source_gain = generate_random_source_signal(
                                        base_config["base_sampling_rate"],
                                        total_duration)
    else:
        random_file_path = random.choice(base_config["anechoic_samples"])
        source_signal, _ = librosa.load(random_file_path,
                                        sr=base_config["base_sampling_rate"])
        source_gain = 1
        

    return {
        "room_dims": room_dims,
        "source_x": source_coordinates[0],
        "source_y": source_coordinates[1],
        "source_coordinates": source_coordinates,
        "mic_coordinates": mic_coordinates,
        "mic_delays": mic_delays,
        "mic_gains": mic_gains,
        "mic_sampling_rates": mic_sampling_rates,
        "tdoa": tdoa,
        "normalized_tdoa": normalized_tdoa,
        "sr": base_config["base_sampling_rate"],
        "source_signal": source_signal,
        "source_gain": source_gain,
        "trim_beginning": trim_beginning
    }


def _compute_tdoa(source_coordinates, mic_coordinates):
    mic_distance = compute_distance(mic_coordinates[0], mic_coordinates[1])
    tdoa = compute_tdoa(source_coordinates, mic_coordinates)
    normalized_tdoa = normalize_tdoa(tdoa, mic_distance)

    return tdoa, normalized_tdoa


def _trim_recorded_signals(signals, num_output_samples, mic_delays, sr, trim_beginning):
    """Trim beginning and end of signals not to have a silence in the beginning,
    which might make the delay detection too easy 
    """
    max_delay = max(mic_delays)
    max_delay_in_samples = int(max_delay*sr)

    # Remove silence in the beginning of signals which might make
    # Detecting the delays "too easy", then truncate to input size
    if trim_beginning:
        signals = signals[:, max_delay_in_samples:]
    
    signals = signals[:, :num_output_samples]

    return signals
