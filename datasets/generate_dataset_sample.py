"""
Generate simulated recordings from an environment
containing two microphones and a source
"""

import pyroomacoustics as pra
import os

from pyroomasync import ConnectedShoeBox, simulate

from datasets.logger import save_signals
from datasets.math_utils import compute_distance, compute_tdoa, normalize_tdoa
from datasets.generate_random_configs import (
    generate_random_points,
    generate_random_microphone_pair_coordinates,
    generate_random_sampling_rate,
    generate_random_source_signal,
    generate_random_speech_signal,
    generate_random_delay,
    generate_random_gain
)


def generate_dataset_sample_config(base_config):
    room_dims = base_config["room_dims"]
    mic_coordinates = base_config["mic_coordinates"]
    mic_distance = base_config["mic_distance"]
    sr = base_config["base_sampling_rate"]

    if not mic_coordinates:
        mic_coordinates = generate_random_microphone_pair_coordinates(
                                                    room_dims, mic_distance)

    source_coordinates = generate_random_points(room_dims, 1)[0]
                                            
    mic_delays = [
        generate_random_delay(*base_config["mic_delay_ranges"][0]),
        generate_random_delay(*base_config["mic_delay_ranges"][1])
    ]
    mic_sampling_rates = [
        generate_random_sampling_rate(*base_config["mic_sampling_rate_ranges"][0]),
        generate_random_sampling_rate(*base_config["mic_sampling_rate_ranges"][1])
    ]
    mic_gains = [
        generate_random_gain(*base_config["mic_gain_ranges"][0]),
        generate_random_gain(*base_config["mic_gain_ranges"][1])
    ]
    
    # Make signal's duration bigger to trim silent beginning
    max_delay = max(mic_delays)
    total_duration = base_config["sample_duration_in_secs"] + max_delay
    if "speech_samples" in base_config:
        source_signal, source_gain = generate_random_speech_signal(
                                        total_duration, sr, base_config["speech_samples"])
    else:
        source_signal, source_gain = generate_random_source_signal(
                                                    sr, total_duration)

        

    tdoa, normalized_tdoa = _compute_tdoa(source_coordinates, mic_coordinates)

    return {
        "room_dims": room_dims,
        "source_coordinates": source_coordinates,
        "mic_coordinates": mic_coordinates,
        "mic_delays": mic_delays,
        "mic_gains": mic_gains,
        "mic_sampling_rates": mic_sampling_rates,
        "tdoa": tdoa,
        "normalized_tdoa": normalized_tdoa,
        "sr": sr,
        "source_signal": source_signal,
        "source_gain": source_gain,
        "trim_beginning": base_config["trim_beginning"],
        "room_absorption": float(base_config["room_absorption"])
    }


def generate_dataset_sample(config, log_melspectrogram=False):
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

    mic_delays = [delay for delay in mic_delays]

    room = ConnectedShoeBox(sample_config["room_dims"],
                            fs=base_sr,
                            materials=pra.Material(sample_config["room_absorption"]))

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
