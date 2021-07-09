from catalyst.dl import FunctionalMetricCallback
from catalyst.callbacks import CheckpointCallback

from neural_tdoa.metrics import average_rms_error


def make_callbacks(log_dir="logs/"):
    callbacks = [
        CheckpointCallback(logdir=log_dir),
        FunctionalMetricCallback(
            "model_output", "targets",
            average_rms_error,
            "rms_error"
        )
    ]
    
    return callbacks
