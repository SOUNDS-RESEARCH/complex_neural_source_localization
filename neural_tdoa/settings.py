from datasets.settings import SR

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
