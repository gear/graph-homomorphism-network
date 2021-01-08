import pytest
import numpy as np
from ghc.data_utils import to_onehot


test_pairs = [(np.array([1,3,2]), np.array([[0,1,0,0],
                                            [0,0,0,1],
                                            [0,0,1,0]])),
              (np.array([4]), np.array([0,0,0,0,1])),
             ]
@pytest.mark.parametrize("arr, one_hot", test_pairs)
def test_one_hot(arr, one_hot):
    assert np.all(to_onehot(arr) == one_hot)