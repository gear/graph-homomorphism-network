import networkx as nx 
from itertools import combinations
from ghc.data_utils import nx2homg


def tree_list(size=6, num_loops=0):
    """Generate nonisomorphic trees up to size `size`."""
    t_list = [tree for i in range(2, size+1) for tree in \
                       nx.generators.nonisomorphic_trees(i)]
    if num_loops > 0:
        t_list = add_loops(t_list, num_loops)
    return t_list


def cycle_list(size=6, num_loops=0):
    """Generate undirected cycles up to size `size`. Parallel
    edges are not allowed."""
    c_list = [nx.generators.cycle_graph(i) for i in range(2, size+1)]
    if num_loops > 0:
        c_list = add_loops(c_list, num_loops)
    return c_list

    
def path_list(size=6, num_loops=0):
    """Generate undirected paths up to size `size`. Parallel
    edges are not allowed."""
    p_list = [nx.generators.path_graph(i) for i in range(2, size+1)]
    if num_loops > 0:
        p_list = add_loops(p_list, num_loops)
    return p_list


def add_loops(graph_lists, num_loops):
    g_with_loops = []
    for g in graph_lists:
        for loop_indices in combinations(g.nodes(), num_loops):
            new_graph = g.copy()
            for i in loop_indices:
                new_graph.add_edge(i,i)
            g_with_loops.append(new_graph)
    g_list = graph_lists + g_with_loops
    return g_list


def hom_profile(size=5):
    """Return a custom homomorphism profile.
    Tree to size 5, cycle to size 5 by default."""
    single_vertex = nx.Graph()
    single_vertex.add_node(0)
    tree_list = [tree for i in range(2,size+1) for tree in \
                    nx.generators.nonisomorphic_trees(i)]
    cycle_list = [nx.generators.cycle_graph(i) for i in range(3, size+1)]
    f_list = [single_vertex]
    f_list.extend(tree_list)
    f_list.extend(cycle_list)
    return f_list
