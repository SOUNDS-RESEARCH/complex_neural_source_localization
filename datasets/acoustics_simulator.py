import pyroomacoustics as pra

from pyroomasync import ConnectedShoeBox, simulate


def simulate_microphone_signals(config):
    """Simulate sound propagation from a sound source to a pair of microphones 

    Args:
        config (dict): Dictionary containing the following keys:
                        - room_dims
                        - sr
                        - anechoic
                        - mic_coordinates
                        - mic_delays
                        - source_coordinates
                        - source_signal

    Returns:
        numpy.array: matrix containing one microphone signal per row
    """

    if config["anechoic"]:
        room = ConnectedShoeBox(config["room_dims"],
                                fs=config["sr"],
                                max_order=0)
    else:
        room = ConnectedShoeBox(config["room_dims"],
                                fs=config["sr"])

    room.add_microphone_array(config["mic_coordinates"],
                              delay=config["mic_delays"],
                              fs=config["mic_sampling_rates"],
                              gain=config["mic_gains"])
    room.add_source(config["source_coordinates"], config["source_signal"])
    signals = simulate(room)
    
    return signals
