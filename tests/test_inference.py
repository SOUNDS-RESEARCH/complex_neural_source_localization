import torch

from complex_neural_source_localization.trainer import TdoaCrnn
from datasets.dataset import TdoaDataset


def test_inference():
    model = TdoaCrnn()
    

    weights = _load_weights()
    model.load_state_dict(weights)
    model.eval()

    dataset = TdoaDataset()

    sample = dataset[0]
    target = sample[1]

    model_output = model(sample[0].unsqueeze(0))


def _load_weights():
    weights_with_prefix = torch.load("tests/fixtures/weights.ckpt",
                                     map_location=torch.device('cpu'))["state_dict"]
    weights = {
        key.replace("model.", ""): value for key, value in weights_with_prefix.items()
    }
    return weights
