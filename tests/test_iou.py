import torch

from yoho.metrics.iou import scalar_iou, compute_iou


def test_iou_no_intersect():
    seg1 = (0, 2)
    seg2 = (3, 4)

    assert scalar_iou(seg1, seg2) == 0


def test_iou_contained():
    seg1 = (0, 2)
    seg2 = (0, 4)

    assert scalar_iou(seg1, seg2) == .5


def test_iou_equal():
    seg1 = (5, 7)
    seg2 = (5, 7)

    assert scalar_iou(seg1, seg2) == 1


def test_batch_iou():
    batch_1 = torch.Tensor([
        [0, 2],
        [0, 2],
        [5, 7]
    ])

    batch_2 = torch.Tensor([
        [3, 4],
        [0, 4],
        [5, 7]
    ])

    ious = compute_iou(batch_1, batch_2)

    assert torch.equal(
        ious,
        torch.Tensor([0, .5, 1])
    )


def test_batch_iou_zeros():
    batch_1 = torch.zeros((3, 2))
    batch_2 = torch.zeros((3, 2))

    ious = compute_iou(batch_1, batch_2)

    assert torch.equal(
        ious,
        torch.zeros(3)
    )


def test_batch_iou_no_intersection():
    batch_1 = torch.Tensor([
        [2, 1],  # (2.5, 3.5)
        [2, 2],  # (1, 3)
        [2, 3]  # (0.5, 3.5)
    ])

    batch_2 = torch.Tensor([
        [1, 1], # (0.5, 1.5)
        [4, 1],  # (3.5, 4.5)
        [5, 1]  # (4.5, 5.5)
    ])

    expected = torch.Tensor([0, 0, 0])

    ious = compute_iou(batch_1, batch_2)

    assert torch.equal(
        ious,
        expected
    )


def test_batch_iou_intersection():
    batch_1 = torch.Tensor([
        [2, 1],  # (1.5, 2.5)
        [2, 2],  # (1, 3)
        [2, 3]  # (0.5, 3.5)
    ])

    batch_2 = torch.Tensor([
        [1.5, 1],  # (1.0, 2.0)
        [4, 1],  # (3.5, 4.5)
        [2, 3]  # (0.5, 3.5)
    ])

    expected = torch.Tensor([1/3, 0, 1])

    ious = compute_iou(batch_1, batch_2)

    assert torch.equal(
        ious,
        expected
    )