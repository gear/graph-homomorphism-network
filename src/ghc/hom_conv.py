import argparse
import torch as T 
import torch.nn as nn 
import torch.nn.functional as nnf 
from utils import load_data, load_tud_data 
import networkx as nx
from collections.abc import Iterable
import numpy as np


class HomConv(nn.Module):
    """Homomorphism convolution layer (F,phi).
    F: Test graph, must be tree.
    fdim: Dimensionality of the vertex features in graph G
    dropout: Dropout rate  
    """

    def __init__(self, F, fdim):
        super().__init__()
        self.weight = nn.Parameter(T.ones(fdim, fdim))
        self.bias = nn.Parameter(T.zeros(fdim))
        self.F = F 
        self.fdim = fdim
    
    def weights_init(self):
        nn.init.eye_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, G, X=None):
        # TODO: density? density for others, raw count for V and E?
        n = G.number_of_nodes()
        nG = G.nodes()
        def rec(x, p):
            if X is None:
                hom_x = T.ones(n).reshape(1,-1).float()
            else:
                hom_x = T.tensor(X).transpose(0,1).float()
            hom_x = T.matmul(self.weight, hom_x).transpose(0,1) + self.bias
            hom_x = nnf.relu(hom_x).transpose(0,1)
            for y in self.F.neighbors(x):
                if y == p:
                    continue
                hom_y = rec(y, x)
                aux = T.tensor([T.sum(hom_y[:, list(G.neighbors(a))]) for a in nG])
                hom_x = hom_x * aux
                if len(hom_x.size()) == 1:
                    hom_x = hom_x.reshape(1,-1)
            return hom_x
        return T.sum(rec(0,-1))

    def test_init(self):
        nn.init.eye_(self.weight)
        nn.init.zeros_(self.bias)


class HNet(nn.Module):
    """Homomorphism Network. This network is restricted to trees.
    This network only contains one layer of homomorphism convolution
    since its expressive power lies at its width.
    """
    def __init__(self, fdim, nclass, hdim=16, max_tree_size=6):
        super().__init__()
        self.hom_conv_modules = []
        self.Fs = [nx.generators.empty_graph(1)]
        for i in range(2, max_tree_size+1):
            self.Fs.extend(nx.generators.nonisomorphic_trees(i))
        for F in self.Fs:
            self.hom_conv_modules.append(HomConv(F, fdim))
        num_F = len(self.Fs)
        self.linear1 = nn.Linear(num_F, hdim) 
        self.linear2 = nn.Linear(hdim, nclass)
        self.bn = nn.BatchNorm1d(num_F)

    def weights_init(self):
        for m in self.hom_conv_modules:
            m.weights_init()

    def hom(self, G, X=None):
        return [m(G, X) for m in self.hom_conv_modules]

    def forward(self, G, X=None):
        if isinstance(G, list):
            emb = T.tensor([self.hom(g, x) for g, x in zip(G, X)])
        else:
            emb = T.tensor(self.hom(G, X)).unsqueeze(0)
        emb = self.bn(emb)
        emb = nnf.relu(self.linear1(emb))
        emb = self.linear2(emb)
        return emb

    def test_init(self):
        """Init weights to identity and bias to zero"""
        for m in self.hom_conv_modules:
            m.test_init()
        

def test_HNet():
    net = HNet(1, 2)
    net.weights_init()
    net.train()
    F = nx.Graph()
    F.add_edge(0, 1)
    G = nx.random_graphs.erdos_renyi_graph(20,0.5)
    print(net([G,G.copy()]))  # Single instance will make batchnorm complains
    net.eval()
    print(net([G]))


def test_HomConv():
    F = nx.Graph()
    F.add_edge(0, 1)
    G = nx.random_graphs.erdos_renyi_graph(20,0.5)
    hom_conv = HomConv(F, 1)
    hom_conv.weights_init()
    hom_conv.train()
    print(hom_conv(G))
    hom_conv = HomConv(F, 3)
    hom_conv.weights_init()
    hom_conv.train()
    X = np.random.randn(G.number_of_nodes(), 3)
    print(hom_conv(G,X))


if __name__ == "__main__":
    test_HomConv()
    #test_HNet()

