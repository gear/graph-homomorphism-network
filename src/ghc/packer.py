"""
This file is meant to use in development only.
"""
import torch_geometric.datasets as dt
import pickle as p
import numpy as np
import os.path as osp
import argparse

from torch_geometric.utils import to_networkx

def gin_cv(dname, path):
    """Read and save cross validation indices from GIN repository."""
    cv = []
    for i in range(1, 11):
        test_idx = np.loadtxt(osp.join(path, "test_idx-{}.txt".format(i)),
                              dtype=int)
        train_idx = np.loadtxt(osp.join(path, "train_idx-{}.txt".format(i)),
                              dtype=int)
        cv.append((train_idx, test_idx))
    p.dump(cv, open(dname+".cv", "wb"))
        

def pack(dname):
    """This function is meant to use in development. It uses torch_geometric 
    TU data downloader and packs the datasets for later use."""
    raw_data = dt.TUDataset("~/data/"+dname, dname, use_node_attr=True)
    try:
        X = np.array([g.x.numpy() for g in raw_data])
    except:
        print("No X")
    y = np.array([g.y.numpy()[0] for g in raw_data])
    graphs = [to_networkx(g).to_undirected() for g in raw_data]
    p.dump(graphs, open(dname+".graph", "wb"))
    p.dump(y, open(dname+".y", "wb"))
    try:
        p.dump(X, open(dname+".X", "wb"))
    except:
        pass

# TODO: Dev for QM datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dname")
    parser.add_argument("--path", default=None)
    args = parser.parse_args() 
    if args.path: 
        gin_cv(args.dname, args.path) 
    else:
        pack(args.dname)
