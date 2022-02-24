import torch

from complex_neural_source_localization.dataset import create_activity_mask


def test_create_activity_mask():
    start, end = (0.35, 0.7)
    max_duration = 1.0
    num_cells = 10

    mask, (start_cell, end_cell) = create_activity_mask(start, end, max_duration, num_cells)

    assert (mask == torch.Tensor([False, False, False, True, True, True, True, False, False, False])).all()
    assert start_cell == 3 and end_cell == 7
