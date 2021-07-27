from neural_tdoa.settings import BASE_MODEL_CONFIG, BASE_TRAINING_CONFIG
from datasets.settings import BASE_DATASET_CONFIG


BASE_EXPERIMENT_CONFIG = {
    "dataset_config": BASE_DATASET_CONFIG,
    "model_config": BASE_MODEL_CONFIG,
    "training_config": BASE_TRAINING_CONFIG,
    "log_dir": "tests/temp/logs"
}