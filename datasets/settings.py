SR = 48000
SAMPLE_DURATION_IN_SECS = 2
NUM_SAMPLES = 100

# 15sqm room with 3m height
ROOM_DIMS = [5, 3, 3]

# The source won't have a fixed position: It'll move around.
N_SOURCES = 1
SOURCE_HEIGHT = 1

MIC_POSITIONS = [
    [1, 1, 1],
    [2, 2, 1]
]
N_MICS = len(MIC_POSITIONS)

DEFAULT_OUTPUT_DATASET_DIR = "generated_dataset"