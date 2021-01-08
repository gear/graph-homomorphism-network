import pytest
import networkx as nx
from ghc.hom_utils import tree_list, cycle_list, path_list, add_loops


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



"""
@pytest.fixture
def small_er():
    '''Small Erdos-Renyi graph with 20 nodes.'''
    g = utils.load_graph("./data/test/er_10_0.5.graph")
    return g

def test_load_graph(small_er):
    assert small_er.number_of_nodes() == 20
    assert small_er.number_of_edges() == 91

def test_simple():
    '''Simple assertion.'''
    assert "homomorphism".capitalize() == "Homomorphism" 
def test_raise():
    '''Check if a certain exception is raised.'''
    with pytest.raises(TypeError):
        assert "1" + 1 == 2
@pytest.fixture
def empty_string():
    '''An empty string'''
    return ""
@pytest.fixture
def some_string():
    '''An exclaimation'''
    return "Graph homomorphism is fascinating!"
def test_with_fixture1(empty_string, some_string):
    assert len(empty_string) == 0
    assert len(some_string) == 34
def test_with_fixture2(some_string):
    print(some_string)
    assert some_string.capitalize() == some_string
"""