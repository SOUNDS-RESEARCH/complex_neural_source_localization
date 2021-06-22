SPEED_OF_SOUND = 343.0
SR = 48000
SAMPLE_DURATION_IN_SECS = 2
N_SAMPLES = 100

# 15sqm room with 3m height
ROOM_DIMS = [5, 3, 3]

MIC_POSITIONS = [
    [1, 1, 1],
    [2, 2, 1]
]
N_MICS = len(MIC_POSITIONS)

DEFAULT_OUTPUT_DATASET_DIR = "generated_dataset"