import torch

from neural_tdoa.model import TdoaCrnn10
from datasets.dataset import TdoaDataset


def test_inference():
    model = TdoaCrnn10()
    
    weights = torch.load("tests/fixtures/weights.pth",
                         map_location=torch.device('cpu'))
    model.load_state_dict(weights["model_state_dict"])

    dataset = TdoaDataset()

    sample = dataset[6]
    target = sample["targets"]

    model_output = model(sample["signals"].unsqueeze(0))

    breakpoint()