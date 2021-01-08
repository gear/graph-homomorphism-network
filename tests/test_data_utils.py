import pytest
from ghc import data_utils

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