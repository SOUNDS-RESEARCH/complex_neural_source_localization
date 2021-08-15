import numpy as np
import pyroomacoustics as pra

from pyroomacoustics.doa.srp import SRP

from pyroomasync.settings import SPEED_OF_SOUND

NFFT = 1024


def srp_phat(simulation_results, room, nfft=NFFT):

    microphone_positions = np.array(room.microphones.get_positions()).T

    sources = room.sources
    fs = room.fs

    srp = SRP(
        microphone_positions,
        fs,
        nfft,
        c=SPEED_OF_SOUND,
        num_src=len(sources),
        mode='near'
    )

    # Compute the STFT frames needed
    simulation_results_stft = np.array(
        [
            pra.transform.stft.analysis(signal, nfft, nfft // 2).T
            for signal in simulation_results
        ]
    )

    srp.locate_sources(simulation_results_stft)

    return srp
