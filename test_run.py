from ghc.homomorphism import hom 
from ghc.utils.data import load_data
from ghc.utils.hom import tree_list

graphs, X, y = load_data("MUTAG", "./data")
trees = tree_list(4, num_loops=1)

# right hom
print(hom(graphs[0], trees[-1]))