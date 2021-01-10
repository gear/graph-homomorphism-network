from nn import MLP
import torch
import pytest


def test_mlp():
    model = MLP(4, 2, [2,3,4], dp=0.2)
    x = torch.tensor([[1,0,0,0],
                      [0,1,0,0],
                      [0,0,1,0]]).float()
    assert model(x).size() == (3, 2)
