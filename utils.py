import pickle as pkl
import networkx as nx
import numpy as np
import random
import torch
import os

from sklearn.model_selection import StratifiedKFold

try:
    from graph_tool.all import Graph as gtGraph
except:
    print("Please install graph-tool to run subgraph density. "\
          "Only pre-computed vectors are available without graph-tool")

try:
    from homlib import Graph as hlGraph
except:
    print("Please install homlib graph library for fast tree homomorphism.")


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
    originally suggested by Takanori Maehara (@spagetti-source)"""
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


def load_du_data(datasset):
    """Utility function to load other datasets for graph 
    classification besides the one provided by GIN.
    """
    g_list = []
    dataf = os.path.dirname(os.path.abspath(__file__))+"/data"
    # Read graph structure
    with open('{}/{}/{}_A.txt'.format(dataf,dataset,dataset), 'r') as gf:

##############################################################
### Copied from https://github.com/weihua916/powerful-gnns ###
##############################################################
def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''
    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    dataf = os.path.dirname(os.path.abspath(__file__))+"/data"
    with open('{}/{}/{}.txt'.format(dataf,dataset,dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]],\
                                np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)),\
                        [tag2index[tag] for tag in g.node_tags]] = 1

    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)


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


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0
##############################################################
##############################################################
##############################################################
