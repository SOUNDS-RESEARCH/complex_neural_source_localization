SPEED_OF_SOUND = 343.0
SR = 48000
SAMPLE_DURATION_IN_SECS = 2
N_SAMPLES = 100

# 15sqm room with 3m height
ROOM_DIMS = [5, 3, 3]

MIC_POSITIONS = [
    [1.01, 1, 1],
    [1, 1, 1]
]
MIC_SAMPLING_RATES = [SR, SR - 10]
MIC_DELAYS = [0, 1e-2]

N_MICS = len(MIC_POSITIONS)

DEFAULT_OUTPUT_DATASET_DIR = "generated_dataset"

BASE_DATASET_CONFIG = {
    "base_sampling_rate": SR,
    "sample_duration_in_secs": SAMPLE_DURATION_IN_SECS,
    "n_training_samples": N_SAMPLES,
    "room_dims": ROOM_DIMS,
    "mic_coordinates": MIC_POSITIONS,
    "mic_sampling_rates": MIC_SAMPLING_RATES,
    "mic_delays": MIC_DELAYS,
    "dataset_dir": DEFAULT_OUTPUT_DATASET_DIR
}