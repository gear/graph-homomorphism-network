from collections import defaultdict
import pickle as pkl
import networkx as nx
import numpy as np
import random
import torch
import os
import random
from sklearn.model_selection import StratifiedKFold
from homlib import Graph as hlGraph


ALL_DATA = ["MUTAG", "PTC_MR", "IMDB-BINARY", "IMDB-MULTI", "NCI1", "PROTEINS",
            "REDDIT-BINARY", "REDDIT-MULTI-5K", "REDDIT-MULTI-12K", "COLLAB", "DD",
            "ENZYMES", "NCI109", "BZR", "COX2", "BZR_MD", "COX2_MD"]


def to_onehot(X, nmax=None):
    """Convert a 1d numpy array to 2d one hot."""
    if nmax is None:
        nmax = X.max()+1
    oh = np.zeros((X.size, nmax))
    oh[np.arange(X.size), X] = 1
    return oh


def save_precompute(X, dataset, hom_type, hom_size):
    dataf = os.path.dirname(os.path.abspath(__file__))+"/data"
    tmp_str = "{}/{}/{}_{}_{}.pkl"
    with open(tmp_str.format(dataf,dataset,dataset,hom_type,hom_size), 
              'wb') as f:
        pkl.dump(X, f)


def load_precompute(dataset, hom_type, hom_size):
    dataf = os.path.dirname(os.path.abspath(__file__))+"/data"
    tmp_str = "{}/{}/{}_{}_{}.pkl"
    try:
        with open(tmp_str.format(dataf,dataset,dataset,hom_type,hom_size), 
                  'rb') as f:
            X = pkl.load(f)
    except:
        X = []
    return X
    

def nx2gt(nxg):
    """Simple function to convert s2v to graph-tool graph."""
    gt = gtGraph(directed=nxg.is_directed())
    gt.add_edge_list(nxg.edges())
    return gt


def nx2homg(nxg):
    """Convert nx graph to homlib graph format. Only 
    undirected graphs are supported. 
    originally suggested by Takanori Maehara (@spagetti-source).
    Note: This function expects nxg to have consecutive integer index."""
    n = nxg.number_of_nodes()
    G = hlGraph(n)
    for (u, v) in nxg.edges():
        G.addEdge(u,v)
    return G


def tree_list(size=6, to_homlib=False):
    """Generate nonisomorphic trees up to size `size`."""
    t_list = [tree for i in range(2,size+1) for tree in \
                       nx.generators.nonisomorphic_trees(i)]
    if to_homlib:
        t_list = [nx2homg(t) for t in t_list]
    return t_list


def cycle_list(size=6, to_homlib=False):
    """Generate undirected cycles up to size `size`. Parallel
    edges are not allowed."""
    c_list = [nx.generators.cycle_graph(i) for i in range(2,size+1)]
    if to_homlib:
        c_list = [nx2homg(c) for c in c_list]
    return c_list


def hom_profile(size=5):
    """Return a custom homomorphism profile.
    Tree to size 5, cycle to size 5 by default."""
    single_vertex = nx.Graph()
    single_vertex.add_node(0)
    tree_list = [tree for i in range(2,size+1) for tree in \
                    nx.generators.nonisomorphic_trees(i)]
    cycle_list = [nx.generators.cycle_graph(i) for i in range(3,size+1)]
    f_list = [single_vertex]
    f_list.extend(tree_list)
    f_list.extend(cycle_list)
    return f_list
    

def path_list(size=6, to_homlib=False):
    """Generate undirected paths up to size `size`. Parallel
    edges are not allowed."""
    p_list = [nx.generators.path_graph(i) for i in range(2,size+1)]
    if to_homlib:
        p_list = [nx2homg(p) for p in p_list]
    return p_list


def graph_type(g):
    if g.__module__ == 'homlib':
        return 'hl'
    elif g.__module__ == 'networkx.classes.graph':
        return 'nx'
    else:
        raise TypeError("Unsupported graph type: {}".format(str(g)))
    #TODO(N): Add for graph-tool type if needed.


def _swap_edges(g, num_swap):
    edges = list(g.edges)
    nodes = list(g.nodes)
    upper = nodes[:int(g.number_of_nodes()/2)]
    lower = nodes[int(g.number_of_nodes()/2):]
    to_change = [random.choice(edges) for _ in range(num_swap)]
    g.remove_edges_from(to_change)
    for _ in range(num_swap):
        u, v = 0, 0
        if random.random() > 0.5:
            sampler = upper
        else:
            sampler = lower
        while u == v:
            u = random.choice(sampler)
            v = random.choice(sampler)
        g.add_edge(u,v)
    return g


def gen_bipartite(num_graphs=200, perm_frac=0.0, p=0.2):
    """Generate bipartite and non-bipartite graphs."""
    bipartites = []
    nonbipartites = []
    for i in range(num_graphs):
        num_nodes = np.random.randint(40,101)
        g = nx.bipartite.generators.random_graph(num_nodes,num_nodes, p)
        if perm_frac > 0:
            num_swap = int(perm_frac * g.number_of_edges())
            g = _swap_edges(g, num_swap)
        bipartites.append(g)
        num_nodes = np.random.randint(40,101)
        g = nx.generators.erdos_renyi_graph(2*num_nodes, p/2)
        nonbipartites.append(g)  # Not 100% fix later

    g_list = []
    for i, g in enumerate(bipartites+nonbipartites):
        g = S2VGraph(g, y[i], node_tags=None, 
                     node_features=None, graph_feature=None)
        g_list.append(g)
    nclass = 2
    return g_list, nclass
    

def load_pickle(dname, root_dir="./data/"):
    """Load datasets"""
    X = None 
    y = None 
    graphs = None 
    name = root_dir+dname
    graphs = pkl.load(open(name+".graph", "rb"))
    y = pkl.load(open(name+".y", "rb"))
    if os.path.exists(name+".X"):
        X = pkl.load(open(name+".X", "rb"))
    return graphs, X, y


def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list