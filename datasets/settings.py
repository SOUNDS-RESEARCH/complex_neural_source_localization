SPEED_OF_SOUND = 343.0
SR = 48000
SAMPLE_DURATION_IN_SECS = 2
N_TRAIN_SAMPLES = 100
N_VALIDATION_SAMPLES = 50

# 15sqm room with 3m height
ROOM_DIMS = [5, 3, 3]

MIC_POSITIONS = [
    [1.01, 1, 1],
    [1, 1, 1]
]
MIC_0_SAMPLING_RATE = SR
MIC_1_SAMPLING_RATE_RANGE = [SR - 10, SR] # From 47900 t0 48000

MIC_0_DELAY = 0
MIC_1_DELAY_RANGE = [0, 10] # From 0 to 10 milliseconds

N_MICS = len(MIC_POSITIONS)

DEFAULT_TRAINING_DATASET_DIR = "generated_dataset"
DEFAULT_VALIDATION_DATASET_DIR = "validation_dataset"

BASE_DATASET_CONFIG = {
    "base_sampling_rate": SR,
    "sample_duration_in_secs": SAMPLE_DURATION_IN_SECS,
    "n_training_samples": N_TRAIN_SAMPLES,
    "n_validation_samples": N_VALIDATION_SAMPLES,
    "room_dims": ROOM_DIMS,
    "mic_coordinates": MIC_POSITIONS,
    "mic_0_sampling_rate": MIC_0_SAMPLING_RATE,
    "mic_1_sampling_rate_range": MIC_1_SAMPLING_RATE_RANGE,
    "mic_0_delay": MIC_0_DELAY,
    "mic_1_delay_range": MIC_1_DELAY_RANGE,
    "training_dataset_dir": DEFAULT_TRAINING_DATASET_DIR,
    "validation_dataset_dir": DEFAULT_VALIDATION_DATASET_DIR
}