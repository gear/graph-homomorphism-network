import torch as T
import torch.nn.functional as nnf
import networkx as nx


def hom_tree(F, G):
    """F must be tree."""
    nG = G.number_of_nodes()
    nGl = G.nodes()
    def rec(x, p):
        hom_x = T.ones(nG).float()
        for y in F.neighbors(x):
            if y == p:
                continue
            hom_y = rec(y, x)  # No cycles
            hom_x = hom_x * T.tensor([T.sum(hom_y[list(G.neighbors(a))]) for a in nGl])
        return hom_x
    return T.sum(rec(0,-1))


def hom_tree_phi(F, G):
    """F must be tree, phi is learnable."""
    n = G.number_of_nodes()
    phi = T.ones(1, 1, requires_grad=True)
    nG = G.nodes()
    def rec(x, p):
        hom_x = T.ones(n).float()
        for y in F.neighbors(x):
            if y == p:
                continue
            hom_y = rec(y, x).view(-1)  # No cycles
            hom_x = hom_x * T.tensor([T.sum(hom_y[list(G.neighbors(a))]) for a in nG])
        return nnf.relu(T.matmul(phi, hom_x.view(1,-1)))
    return T.sum(rec(0,-1))
            

def test_hom_tree():
    F = nx.Graph()
    F.add_node(0)
    G = nx.random_graphs.erdos_renyi_graph(20,0.5)
    assert hom_tree(F,G).item() == 20, "hom_tree(F,G) = {}".format(hom_tree(F,G))
    F = nx.Graph()
    F.add_edge(0,1)
    assert hom_tree(F,G) == 2.0 * G.number_of_edges(), "hom_tree(F,G) = {}".format(hom_tree(F,G))
    print("hom_tree works.")


def test_hom_tree_phi():
    F = nx.Graph()
    F.add_edge(0, 1)
    G = nx.random_graphs.erdos_renyi_graph(20,0.5)
    print(hom_tree_phi(F,G))


if __name__ == "__main__":
    test_hom_tree()
    test_hom_tree_phi()