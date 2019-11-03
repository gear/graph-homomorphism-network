from utils import nx2homg, tree_list, cycle_list,\
                  path_list, graph_type
import homlib as hl
import networkx as nx
import numpy as np
from  multiprocessing import Pool


def hom_tree(F, G):
    """Specialized tree homomorphism in Python (serializable).
    By: Takanori Maehara (maehara@prefield.com)
    """
    def rec(x, p):
        hom_x = np.ones(len(G.nodes()), dtype=float)
        for y in F.neighbors(x):
            if y == p:
                continue
            hom_y = rec(y, x)
            aux = [np.sum(hom_y[list(G.neighbors(a))]) for a in G.nodes()]
            hom_x *= np.array(aux)
        return hom_x
    hom_r = rec(0, -1)
    return np.sum(hom_r)


def hom(F, G, f_is_tree=False):
    """Wrapper for the `hom` function in `homlib`
    (https://github.com/spaghetti-source/homlib). 
    If `f_is_tree`, then use the Python implementation of tree. This one is 
    10 times slower than homlib but can be parallelize with `multiprocessing`.
    """
    # Default homomorphism function
    hom_func = hl.hom
    # Check if tree, then change the hom function
    if f_is_tree:
        hom_func = hom_tree
    # Check and convert graph type
    if graph_type(F) == "nx" and not f_is_tree:
        F = nx2homg(F)
    if graph_type(G) == "nx" and not f_is_tree:
        G = nx2homg(G)
    return hom_func(F, G) 


def tree_profile(G, size=6):
    """Run tree homomorphism profile for a single graph G."""
    t_list = tree_list(6, to_homlib=True)
    return [hom(t, G) for t in t_list]


def path_profile(G, size=6):
    """Run tree homomorphism profile for a single graph G."""
    p_list = path_list(6, to_homlib=True)
    return [hom(p, G) for p in p_list]


def cycle_profile(G, size=6):
    """Run tree homomorphism profile for a single graph G."""
    c_list = cycle_list(6, to_homlib=True)
    return [hom(c, G) for c in c_list]
    

def get_hom_profile(f_str):
    if f_str == "tree":
        return tree_profile
    elif f_str == "path":
        return path_profile
    elif f_str == "cycle":
        return cycle_profile
    else:
        raise ValueError("{} is not a valid F class.".format(f_str))