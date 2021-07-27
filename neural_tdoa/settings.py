from datasets.settings import SR

# Training config
BATCH_SIZE = 32
NUM_EPOCHS = 8
LEARNING_RATE = 0.0001

BASE_TRAINING_CONFIG = {
    "batch_size": BATCH_SIZE,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE
}

# NN Model settings
POOL_TYPE = "avg" # "max" | "avg"
POOL_SIZE = (2, 2)
N_OUTPUT_CHANNELS = 512
N_CONV_LAYERS = 4

# Feature params
N_FFT = 1024
N_MELS = 64
WINDOW = "hann"

HOP_LENGTH_IN_SECONDS = 10e-3
HOP_LENGTH = int(HOP_LENGTH_IN_SECONDS * SR)


BASE_MODEL_CONFIG = {
    "feature_type": "stft_magnitude"
}