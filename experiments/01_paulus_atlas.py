import sys
sys.path.append('..')
from sacred import Experiment

ex = Experiment('1-paulus-atlas')

@ex.config
def data_config():
    graphs = "../data/PAULUS25.graph"
    labels = "../data/PAULUS25.y"
    experiment_type = "synthetic"
     
@ex.config
def hom_config():
    from homomorphism import get_hom_profile
    hom_type = "atlas"
    hom_size = 20
    hom_func = get_hom_profile(hom_type)

@ex.automain
def run(graphs, labels, hom_size, hom_func):
    import pickle as pkl
    graphs_nx = pkl.load(open(graphs, "rb"))
    labels_ar = pkl.load(open(labels, "rb"))
    for g, l in zip(graphs_nx, labels_ar):
        print(l, hom_func(g, hom_size))


