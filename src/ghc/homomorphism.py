from ghc.hom_utils import nx2homg, tree_list, cycle_list,\
                          path_list, hom_profile
import homlib as hl
import networkx as nx
import numpy as np
from multiprocessing import Pool


def hom_tree(F, G):
    """Specialized tree homomorphism in Python (serializable).
    Add `indexed` parameter to count for each index individually.
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


def hom_tree_labeled(F, G, node_tags=None):
    """Tree homomorphism with node labels (tags).
    """
    if node_tags is None:
        print("Warning: Missing node tags.")
        return hom_tree(F, G)
    if type(node_tags) is list:
        node_tags = np.array(node_tags)
    def rec(x, p):
        hom_x = node_tags.T.copy()
        for y in F.neighbors(x):
            if y == p:
                continue
            hom_y = rec(y, x)
            aux = [np.sum(hom_y[:,list(G.neighbors(a))]) for a in G.nodes()]
            hom_x *= np.array(aux)
        return hom_x
    hom_r = rec(0, -1)
    return np.sum(hom_r, axis=1)


def hom_tree_explabeled(F, G, node_tags=None, exp=np.e):
    """Tree homomorphism with node labels (tags).
    """
    if node_tags is None:
        print("Warning: Missing node tags.")
        return hom_tree(F, G)
    if type(node_tags) is list:
        node_tags = np.array(node_tags)
    def rec(x, p):
        hom_x = np.power(exp, node_tags.T.copy())
        for y in F.neighbors(x):
            if y == p:
                continue
            hom_y = rec(y, x)
            aux = [np.sum(hom_y[:,list(G.neighbors(a))]) for a in G.nodes()]
            hom_x *= np.array(aux)
        return hom_x
    hom_r = rec(0, -1)
    return np.log(np.sum(hom_r, axis=1))


def hom(F, G, f_is_tree=False, density=False):
    """Wrapper for the `hom` function in `homlib`
    If `f_is_tree`, then use the Python implementation of tree. This one is 
    10 times slower than homlib but can be parallelize with `multiprocessing`.
    """
    # assert graph_type(G) == "nx" and graph_type(F) == "nx", "Invalid type."
    # Default homomorphism function
    hom_func = hl.hom
    # Check if tree, then change the hom function
    if f_is_tree:
        hom_func = hom_tree
    # Check and convert graph type
    if density:
        scaler = 1.0 / (G.number_of_nodes() ** F.number_of_nodes())
    else:
        scaler = 1.0
    if not f_is_tree:
        F = nx2homg(F)
        G = nx2homg(G)
    return hom_func(F, G) * scaler


def atlas_profile(G, size=20, start=0, density=False, **kwargs):
    """Run homomorphism count for each graph in the nx atlas"""
    atlas_list = nx.atlas.graph_atlas_g()[start:start+size]
    return [hom(ga, G, density=density) for ga in atlas_list]


def tree_profile(G, size=6, density=False, **kwargs):
    """Run tree homomorphism profile for a single graph G."""
    t_list = tree_list(size, to_homlib=False)
    return [hom(t, G, density=density) for t in t_list]


def path_profile(G, size=6, density=False, **kwargs):
    """Run tree homomorphism profile for a single graph G."""
    p_list = path_list(size, to_homlib=False)
    return [hom(p, G, density=density) for p in p_list]


def cycle_profile(G, size=6, density=False, **kwargs):
    """Run tree homomorphism profile for a single graph G."""
    c_list = cycle_list(size, to_homlib=False)
    return [hom(c, G, density=density) for c in c_list]


def tree_cycle_profile(G, size=6, density=False, **kwargs):
    """Run profile for both tree and cycle."""
    tree_pf = tree_profile(G, size, density)
    cycle_pf = cycle_profile(G, size, density)
    return tree_pf + cycle_pf


def labeled_tree_profile(G, size=6, node_tags=None, **kwargs):
    """Run profile for labeled trees."""
    t_list = tree_list(size, to_homlib=False)
    hom_list = [hom_tree_labeled(t, G, node_tags) for t in t_list]
    return np.concatenate(hom_list)


def explabeled_tree_profile(G, size=6, node_tags=None, **kwargs):
    """Run profile for exponentially labeled trees."""
    t_list = tree_list(size, to_homlib=False)
    hom_list = [hom_tree_explabeled(t, G, node_tags) for t in t_list]
    return np.concatenate(hom_list)


def homomorphism_profile(G, size=6, node_tags=None, **kwargs):
    """Run profile for exponentially labeled trees."""
    t_list = hom_profile(size)
    hom_list = [hom(t, G) for t in t_list]
    return hom_list


def get_hom_profile(f_str):
    if f_str == "labeled_tree":
        return labeled_tree_profile
    elif f_str == "explabeled_tree":
        return explabeled_tree_profile
    elif f_str == "tree":
        return tree_profile
    elif f_str == "path":
        return path_profile
    elif f_str == "cycle":
        return cycle_profile
    elif f_str == "tree+cycle":
        return tree_cycle_profile
    elif f_str == "atlas":
        return atlas_profile
    else:
        return homomorphism_profile
