import torch

from neural_tdoa.model import TdoaCrnn
from datasets.dataset import TdoaDataset


def test_inference():
    model = TdoaCrnn()
    
    weights = torch.load("tests/fixtures/weights.pth",
                         map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    model.eval()

    dataset = TdoaDataset()

    sample = dataset[0]
    target = sample[1]

    model_output = model(sample[0].unsqueeze(0))
