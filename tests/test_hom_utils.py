import pytest
import networkx as nx
from ghc.utils.hom import tree_list, cycle_list, path_list, add_loops


def empty():
    '''Empty graph'''
    g = nx.Graph()
    return g


def singleton():
    '''Simple triangle graph'''
    g = nx.Graph()
    g.add_nodes_from([0])
    return g


def triangle():
    '''Simple triangle graph'''
    g = nx.Graph()
    g.add_nodes_from([0,1,2])
    g.add_edges_from([(0,1), (1,2), (2,0)])
    return g


def path3():
    '''Simple P3'''
    g = nx.Graph()
    g.add_nodes_from([0,1,2])
    g.add_edges_from([(0,1), (1,2)])
    return g


@pytest.mark.parametrize("g", [empty(), singleton(), triangle(), path3()])
def test_add_single_loop(g):
    g_list = add_loops([g], 1)
    assert len(g_list) == 1 + g.number_of_nodes()
    for i in range(1, 1+g.number_of_nodes()):
        assert g_list[i].number_of_nodes() == g.number_of_nodes()
        assert g_list[i].number_of_edges() == g.number_of_edges() + 1
        assert set(g_list[i].edges()) - set(g.edges()) == set([(i-1, i-1)])


@pytest.mark.parametrize("g", [empty(), singleton(), triangle(), path3()])
def test_add_two_loops(g):
    g_list = add_loops([g], 2)
    expected_num_graphs = 1 + g.number_of_nodes()*(g.number_of_nodes()-1)//2
    assert len(g_list) == expected_num_graphs
    for i in range(1, expected_num_graphs):
        assert g_list[i].number_of_nodes() == g.number_of_nodes()
        assert g_list[i].number_of_edges() == g.number_of_edges() + 2


def test_cycle_list():
    cyclel2 = path_list(size=4, num_loops=1)
    assert 12 == len(cyclel2)


def test_path_list():
    pathl2 = path_list(size=2, num_loops=0)
    assert 1 == len(pathl2)


def test_tree_list():
    treel2 = tree_list(size=2, num_loops=0)
    assert 1 == len(treel2)
    treel2 = tree_list(size=2, num_loops=1)
    assert 3 == len(treel2)
    treel6 = tree_list(size=6, num_loops=0)
    assert 13 == len(treel6)
    treel6 = tree_list(size=6, num_loops=1)
    assert 77 == len(treel6)
