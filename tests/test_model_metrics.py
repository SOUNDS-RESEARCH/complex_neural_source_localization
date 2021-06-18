import torch
from yoho.model.metrics import _compute_ious

EPS = 1e-5


def test_compute_iou():
    predictions = torch.Tensor(
        [
            [0, 1],
            [0.4, 1],
            [0.9, 0.1],
        ]
    )

    target = torch.Tensor(
        [
            0.4,
            0.8,
            1
        ]
    )

    expected_ious = torch.Tensor(
        [
            [0, 0.4],
            [0.5, 0.8],
            [0.9, 0.1]
        ]
    )

    ious = _compute_ious(predictions, target)

    assert (ious - expected_ious).sum() < EPS
