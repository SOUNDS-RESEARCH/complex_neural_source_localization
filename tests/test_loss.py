import torch

from neural_tdoa.metrics import Loss
from neural_tdoa.model import TdoaCrnn10
from datasets.dataset import TdoaDataset


def test_neural_tdoa_loss():

    loss_fn = Loss()
    model = TdoaCrnn10()

    dataset = TdoaDataset()

    sample = dataset[0]
    target = sample[1]

    model_output = model(sample[0].unsqueeze(0))
    
    _ = loss_fn(model_output, torch.Tensor([[target]]))
