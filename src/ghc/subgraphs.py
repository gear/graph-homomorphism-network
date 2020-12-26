from graph_tool.all import motifs
from utils import load_data, nx2gt
import pickle as p 
from time import time
import argparse 
import os


dataset_list = ["MUTAG", "PTC", "PROTEINS", "NCI1", "IMDBMULTI", 
                "IMDBBINARY", "COLLAB", "REDDITBINARY", "REDDITMULTI5K"]

forbiden_list = ["REDDITBINARY", "REDDITMULTI5K", "COLLAB"]

parser = argparse.ArgumentParser(description="Compute and dump subgraphs from a dataset.")
parser.add_argument('--dataset', type=str, default="MUTAG", choices=dataset_list)
parser.add_argument('--dump_folder', type=str, default="/warehouse/subgraph/")


def precompute_pickle(dataset="MUTAG", dump_folder="data/subgraph/"):
    """Compute subgraph and dump the results."""
    start = time()
    glist, _ = load_data(dataset, False)
    dump_folder = dump_folder+dataset+'/'
    if not os.path.exists(dump_folder):
        os.makedirs(dump_folder)
    for gid, s2vg in enumerate(glist):
        g = s2vg.g
        gt = nx2gt(g)
        for i in range(2,gt.num_vertices()+1):
            m, c, vm = motifs(gt, i, return_maps=True)
            with open(dump_folder+"g{}_m_{}.pkl".format(gid, i), 'wb') as f:
                p.dump(m, f)
            with open(dump_folder+"g{}_c_{}.pkl".format(gid, i), 'wb') as f:
                p.dump(c, f)
            with open(dump_folder+"g{}_vm_{}.pkl".format(gid, i), 'wb') as f:
                p.dump(vm, f)
    end = time()
    delta_time = end-start
    return delta_time


def main(dataset, dump_folder):
    if dataset in forbiden_list:
        print("WARNING: Running for {}".format(dataset))
        print("WARNING: It might never finish running...")
    print("Running {} and save at {}".format(dataset, dump_folder))
    t = precompute_pickle(dataset=dataset, dump_folder=dump_folder) 
    print("Time: {:.4f}".format(t))
    return t


if __name__ == "__main__":
    args = parser.parse_args()
    t = main(args.dataset, args.dump_folder)
