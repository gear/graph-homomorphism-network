from collections import defaultdict
import pickle as pkl
import networkx as nx
import numpy as np
import random
import os
import random
from itertools import repeat
from sklearn.model_selection import StratifiedKFold
from homlib import Graph as hlGraph


ALL_DATA = ["MUTAG", "PTC_MR", "IMDB-BINARY", "IMDB-MULTI", "NCI1", "PROTEINS",
            "REDDIT-BINARY", "REDDIT-MULTI-5K", "REDDIT-MULTI-12K", "COLLAB",
            "DD", "ENZYMES", "NCI109", "BZR", "COX2", "BZR_MD", "COX2_MD"]


def to_onehot(y, nmax=None):
    '''Convert a 1d numpy array to 2d one hot.'''
    if y.size == 0:
        return y
    if nmax is None:
        nmax = y.max()+1
    oh = np.zeros((y.size, nmax))
    oh[np.arange(y.size), y] = 1
    return oh


def save_precompute(X, dataset, hom_type, hom_size, dloc):
    dataf = os.path.abspath(dloc)
    tmp_str = "{}/{}_{}_{}.hom"
    with open(tmp_str.format(dataf,dataset,hom_type,hom_size), 'wb') as f:
        pkl.dump(X, f)


def load_precompute(dataset, hom_type, hom_size, dloc):
    dataf = os.path.abspath(dloc)
    tmp_str = "{}/{}_{}_{}.hom"
    with open(tmp_str.format(dataf,dataset,hom_type,hom_size), 'rb') as f:
        X = pkl.load(f)
    return X


def nx2homg(nxg):
    """Convert nx graph to homlib graph format. Only
    undirected graphs are supported.
    Note: This function expects nxg to have consecutive integer index."""
    n = nxg.number_of_nodes()
    G = hlGraph(n)
    for (u, v) in nxg.edges():
        G.addEdge(u,v)
    return G


#def _swap_edges(g, num_swap):
#    edges = list(g.edges)
#    nodes = list(g.nodes)
#    upper = nodes[:int(g.number_of_nodes()/2)]
#    lower = nodes[int(g.number_of_nodes()/2):]
#    to_change = [random.choice(edges) for _ in range(num_swap)]
#    g.remove_edges_from(to_change)
#    for _ in range(num_swap):
#        u, v = 0, 0
#        if random.random() > 0.5:
#            sampler = upper
#        else:
#            sampler = lower
#        while u == v:
#            u = random.choice(sampler)
#            v = random.choice(sampler)
#        g.add_edge(u,v)
#    return g


#def gen_bipartite(num_graphs=200, perm_frac=0.0, p=0.2):
#    """Generate bipartite and non-bipartite graphs."""
#    bipartites = []
#    nonbipartites = []
#    for i in range(num_graphs):
#        num_nodes = np.random.randint(40,101)
#        g = nx.bipartite.generators.random_graph(num_nodes,num_nodes, p)
#        if perm_frac > 0:
#            num_swap = int(perm_frac * g.number_of_edges())
#            g = _swap_edges(g, num_swap)
#        bipartites.append(g)
#        num_nodes = np.random.randint(40,101)
#        g = nx.generators.erdos_renyi_graph(2*num_nodes, p/2)
#        nonbipartites.append(g)  # Not 100% fix later

#    g_list = []
#    for i, g in enumerate(bipartites+nonbipartites):
#        g = S2VGraph(g, y[i], node_tags=None, 
#                     node_features=None, graph_feature=None)
#        g_list.append(g)
#    nclass = 2
#    return g_list, nclass


def load_data(dname, dloc):
    """Load datasets"""
    X = None
    y = None
    graphs = None
    name = os.path.abspath(os.path.join(dloc, dname))
    with open(name+".graph", "rb") as f:
        graphs = pkl.load(f)
    with open(name+".y", "rb") as f:
        y = pkl.load(f)
    if os.path.exists(name+".X"):
        with open(name+".X", "rb") as f:
            X = pkl.load(f)
    return graphs, X, y


def load_folds(dname, dloc):
    """Load preassigned 10-folds splits for each datasets"""
    splits = None
    name = os.path.abspath(os.path.join(dloc, dname))
    with open(name+".folds", "rb") as f:
        splits = pkl.load(f)
    return splits


#def drop_nodes(graphs, X, y, per_graph=2, rate=0.1):
#    generated_graphs = []
#    for i, g in enumerate(graphs):
#        n = g.number_of_nodes()
#        for _ in range(per_graph):
#            ng = g.copy()
#            droplist = np.random.choice(ng.nodes(),
#                                        size=int(rate*n),
#                                        replace=False)
#            ng.remove_nodes_from(droplist)
#            #### Reindexing
#            mapping = dict([(i, j) for j, i in enumerate(ng.nodes())])


def drop_nodes(graph, x, rate=1):
    #### Remove nodes
    n = graph.number_of_nodes()
    ng = graph.copy()
    if type(rate) is float:
        num_drop = int(rate*n)
    else:
        num_drop = rate
    droplist = np.random.choice(ng.nodes(), size=num_drop, replace=False)
    ng.remove_nodes_from(droplist)
    #### Reindex to consecutive integers
    mapping = dict([(i, j) for j, i in enumerate(ng.nodes())])
    ng = nx.relabel_nodes(ng, mapping=mapping)
    newx = x[list(mapping.keys()), :]
    return ng, newx


def augment_data(graphs, X, y, samples_per_graph, rate=1):
    new_graphs = []
    new_X = []
    new_y = [[ny] for dupy in y for ny in repeat(dupy, samples_per_graph)]
    for g, x in zip(graphs, X):
        gen_data = [drop_nodes(g, x, rate) for _ in range(samples_per_graph)]
        gen_graphs, gen_x = zip(*gen_data)
        new_graphs.extend(gen_graphs)
        new_X.extend(gen_x)
    return new_graphs, np.array(new_X, dtype=object), np.array(new_y)
