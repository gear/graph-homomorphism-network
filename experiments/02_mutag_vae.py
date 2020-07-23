import sys
sys.path.append('..')
from sacred import Experiment

ex = Experiment('2-mutag-vae')

@ex.config
def data_config():
    graphs = "../data/MUTAG.graph"
    labels = "../data/MUTAG.y"
    experiment_type = "real-world"
     
@ex.config
def hom_config():
    from homomorphism import get_hom_profile
    hom_type = "tree"
    hom_size = 5
    hom_func = get_hom_profile(hom_type)

@ex.config
def note():
    note = """
            No vertex/edge features are used here. This experiment try to see
            if encoding homomorphism works at least as good as SOTA.
           """

@ex.automain
def run(graphs, labels, hom_size, hom_func):
    import pickle as pkl
    graphs_nx = pkl.load(open(graphs, "rb"))
    labels_ar = pkl.load(open(labels, "rb"))
    for g, l in zip(graphs_nx, labels_ar):
        print(l, hom_func(g, hom_size))


