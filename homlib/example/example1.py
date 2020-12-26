import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))


from homlib import Graph, hom
T = Graph(3)
T.addEdge(0,1)
T.addEdge(1,2)

G = Graph(3)
G.addEdge(0,1)
G.addEdge(1,2)
G.addEdge(2,0)

print(hom(T, G))
