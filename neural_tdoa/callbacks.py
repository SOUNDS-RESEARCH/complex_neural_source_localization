from catalyst.dl import FunctionalMetricCallback
from catalyst.callbacks import CheckpointCallback

from neural_tdoa.metrics.scores import average_l1_distance

def make_callbacks():
    callbacks = [
        CheckpointCallback(logdir="logs/"),
        FunctionalMetricCallback(
            "model_output", "targets",
            average_l1_distance,
            "l1_error"
        )
    ]
    
    return callbacks
