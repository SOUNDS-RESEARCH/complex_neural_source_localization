import torch

from yoho.model.yoho_target import (
    make_yoho_target_from_dcase_2021_annotation)
from yoho.model.yoho_net import YohoNet
from yoho.model.dataset import YohoDataset


def test_yoho_net():

    model = YohoNet()

    dataset = YohoDataset(file_paths=[(
        "tests/fixtures/fold1_room1_mix001.wav",
        "tests/fixtures/fold1_room1_mix001.csv"
    )])

    sample = dataset[0]
    targets = sample["targets"]

    model_output = model(sample["signal"].unsqueeze(0))

    assert list(model_output.keys()) == [
        'full', 'classes',
        'predictors', 'durations',
        'azimuths', 'elevations',
        'confidences'
    ]

    assert model_output["full"].shape == (10, 20)
