import os
import pytest
import numpy as np
from ghc.data_utils import to_onehot, save_precompute, load_precompute


test_pairs = [(np.array([1,3,2]), np.array([[0,1,0,0],
                                            [0,0,0,1],
                                            [0,0,1,0]])),
              (np.array([4]), np.array([0,0,0,0,1])),
             ]
@pytest.mark.parametrize("arr, one_hot", test_pairs)
def test_one_hot(arr, one_hot):
    assert np.all(to_onehot(arr) == one_hot)


def test_save_load():
    X = np.array([[0,1,0,0],
                  [0,0,0,1],
                  [0,0,1,0]])
    dataset = "test"
    hom_type = "hom"
    hom_size = "0"
    dloc = "/tmp/"
    save_precompute(X, dataset, hom_type, hom_size, dloc)
    assert os.path.exists("/tmp/test_hom_0.hom")

    Xp = load_precompute(dataset, hom_type, hom_size, dloc)
    assert np.all(Xp == X)