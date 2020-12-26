import os
import numpy as np
import pickle as pkl
import torch as T 
from torch.utils import data as td
from sklearn.model_selection import StratifiedKFold
from collections import namedtuple


class Dataset(td.Dataset):
    """Custom in-memory dataset to load graphs"""
    def __init__(self, graphs, X, labels, idx):
        self.labels = labels[idx]
        self.graphs = [graphs[i] for i in idx]
        self.X = X[idx]
        self.idx = idx
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.graphs[index], self.X[index]), self.labels[index]

    def original_index(self, index):
        return self.idx[index]


class CVDataset(object):
    """Wrapper class to generate Stratified k-folds."""
    def __init__(self, dname, num_folds=10, root_dir='./data/'):
        graphs, X, y, nclass, fdim = _load_raw(dname, root_dir)
        self.dname = dname
        self.nclass = nclass
        self.fdim = fdim 
        self.graphs = graphs 
        self.X = X
        self.y = y 
        self.nfolds = num_folds 

    def folds(self):
        skf = StratifiedKFold(n_splits=self.nfolds)
        for train_idx, val_idx in skf.split(self.X, self.y):
            tid = T.tensor(train_idx).long()
            vid = T.tensor(val_idx).long()
            tdset = Dataset(self.graphs, self.X, self.y, tid)
            vdset = Dataset(self.graphs, self.X, self.y, vid)
            yield tdset, vdset


def torch_data(dname, num_folds=10):
    """Util function to create CVDataset"""
    dataset = CVDataset(dname)
    return dataset


def _load_raw(dname, root_dir='./data/'):
    """This function load the pre-packed data. 
    Check packer and install torch_geometric to run on other datasets.
    """
    root_dir = os.path.expanduser(root_dir)
    graphs = os.path.join(root_dir, dname+".graph")
    graphs = pkl.load(open(graphs, "rb"))

    X = os.path.join(root_dir, dname+".X")
    X = pkl.load(open(X, "rb"))

    y = os.path.join(root_dir, dname+".y")
    y = pkl.load(open(y, "rb"))

    fdim = X[0].shape[1]
    nclass = len(np.unique(y))

    return graphs, X, y, nclass, fdim


def test_load_raw():
    graphs, X, y, nclass, fdim = _load_raw("MUTAG")
    assert len(graphs) == 188, "Wrong number of graphs"
    assert len(X) == 188, "Wrong number of vertex features"
    assert len(y) == 188, "Wrong number of graphs"
    assert nclass == 2, "Wrong number of class"
    assert fdim == 7, "Wrong vertex features dim"
    print("Test _load_raw passed!")


def test_torch_data():
    dataset = torch_data("MUTAG", num_folds=10)
    for tdset, vdset in dataset.folds():
        _ = tdset[0]
        _ = vdset[0]
    print("Test torch_data passed!")

    
if __name__ == "__main__":
    test_load_raw()
    test_torch_data()