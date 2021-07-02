import torch

from neural_tdoa.models.common.complex import (
    ComplexConv2d, ComplexLinear, complex_relu
)


def test_linear():
    x = torch.rand((1, 1000), dtype=torch.cfloat)

    layer = ComplexLinear(1000, 1000)

    result = layer(x)

    assert result.shape == (1, 1000)


def test_conv_2d():
    x = torch.rand((1, 3, 100, 100), dtype=torch.cfloat)

    layer = ComplexConv2d(3, 64)

    result = layer(x)

    assert result.shape == (1, 64, 100, 100)


def test_complex_relu():
    x = torch.tensor([-1 + 4j, 1 - 3j]) 

    result = complex_relu(x)

    assert (result == torch.tensor([4j, 1])).all()