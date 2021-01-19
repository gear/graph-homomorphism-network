import os
import pytest
import numpy as np
import networkx as nx
from ghc.utils.data import to_onehot, save_precompute, load_precompute,\
                           drop_nodes, augment_data, load_data


test_pairs = [(np.array([1,3,2]), np.array([[0,1,0,0],
                                            [0,0,0,1],
                                            [0,0,1,0]])),
              (np.array([4]), np.array([0,0,0,0,1]))]
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


def test_augment_data():
    graphs, X, y = load_data("MUTAG", "./tests/")
    new_graphs, new_X, new_y = augment_data(graphs, X, y, 2, rate=1)
    for i, g in enumerate(graphs):
        assert g.number_of_nodes() - new_graphs[2*i].number_of_nodes() == 1
        assert g.number_of_nodes() - new_graphs[2*i+1].number_of_nodes() == 1
    for i, yi in enumerate(y):
        assert yi == new_y[2*i] == new_y[2*i+1]
    for i, xi in enumerate(X):
        assert xi.shape[0] == new_X[2*i].shape[0]+1 == new_X[2*i+1].shape[0]+1
        assert xi.shape[1] == new_X[2*i].shape[1] == new_X[2*i+1].shape[1]
