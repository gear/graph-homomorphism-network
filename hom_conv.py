import argparse
import torch as T 
import torch.nn as nn 
import torch.nn.functional as nnf 
from utils import load_data, load_tud_data 
import networkx as nx
from collections.abc import Iterable

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
        nn.init.normal_(self.weight, mean=0, std=1)
        nn.init.zeros_(self.bias)

    def forward(self, G):
        # TODO: density? density for others, raw count for V and E?
        n = G.number_of_nodes()
        nG = G.nodes()
        def rec(x, p):
            hom_x = T.ones(n).float()
            for y in self.F.neighbors(x):
                if y == p:
                    continue
                hom_y = rec(y, x).view(-1)  # No cycles
                hom_x = hom_x * T.tensor([T.sum(hom_y[list(G.neighbors(a))]) for a in nG])
            return nnf.relu(T.matmul(self.weight, hom_x.view(1,-1)) + self.bias)
        return T.sum(rec(0,-1))


class HNet(nn.Module):
    """Homomorphism Network. This network is restricted to trees.
    This network only contains one layer of homomorphism convolution
    since its expressive power lies at its width.
    """
    def __init__(self, fdim, nclass, hdim=16, tree_max_size=6):
        super().__init__()
        self.hom_conv_modules = []
        self.Fs = [nx.generators.empty_graph(1)]
        for i in range(2, tree_max_size+1):
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

    def hom(self, G):
        return [m(G) for m in self.hom_conv_modules]

    def forward(self, G):
        if isinstance(G, Iterable):
            emb = T.tensor([self.hom(g) for g in G])
        else:
            emb = T.tensor(self.hom(G)).unsqueeze(0)
        emb = self.bn(emb)
        emb = nnf.relu(self.linear1(emb))
        emb = self.linear2(emb)
        return emb


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


if __name__ == "__main__":
    #test_HomConv()
    test_HNet()

