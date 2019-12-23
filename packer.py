import torch_geometric.datasets as dt
import pickle as p
import numpy as np
import argparse

from torch_geometric.utils import to_networkx

def pack(dname):
    """This function is meant to use in development. It uses torch_geometric 
    TU data downloader and packs the datasets for later use."""
    raw_data = dt.TUDataset("~/data/"+dname, dname, use_node_attr=True)
    try:
        X = np.array([g.x.numpy() for g in raw_data])
    except:
        print("No X")
    y = np.array([g.y.numpy() for g in raw_data])
    graphs = [to_networkx(g) for g in raw_data]
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
    args = parser.parse_args()
    pack(args.dname)
