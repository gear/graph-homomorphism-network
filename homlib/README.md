# homlib --- An implementation of a Homomorphism Counting Algorithm

## Description

This library computes the number of graph homomophisms, hom(F, G).


This implements Diaz--Serna--Thilikos's dynamic programming algorithm on a nice tree decomposition, where the greedy heuristics is used to find a tree decomposition. The complexity is O(poly(|V(F)|) |V(G)|^{tw(F)+1}). If F is tree, it runs in O(|V(F)||E(G)|).

Josep Diaz, Maria Serna, Dimitrios M. Thilikos (2002): "Counting H-colorings of partial k-trees", Theoretical Computer Science, 281 (2002), 291 – 309.

## Usage

````
from homlib import Graph, hom
T = Graph(3)
T.addEdge(0,1)
T.addEdge(1,2)

G = Graph(3)
G.addEdge(0,1)
G.addEdge(1,2)
G.addEdge(2,0)

print(hom(T, G))
````

## Install

### C++

````
git clone https://github.com/spaghetti-source/homlib/
cd homlib
make
````

### Python

It depends on pybind11 (https://github.com/pybind/pybind11)

````
git clone https://github.com/spaghetti-source/homlib/
pip3 install ./homlib
````

### Note for Anaconda

If `anaconda` environment is used, anaconda `ld` will 
overshadow the system's linker, which might fail to build.
To fix it, temporarily change anaconda's linker to `ld_old`
and change it back after installing.
```
cd /home/user/$username$/anaconda3/envs/$envname$/compiler_compat/
mv ld ld_old
```
also
```
cd /home/user/$username$/anaconda3/compiler_compat/
mv ld ld_old
```
Similar issue can be found [here](https://github.com/pytorch/pytorch/issues/16683).

## Uninstall

### C++

````
rm homlib
````

### Python

````
pip3 uninstall homlib
rm homlib
````

## Author

Takanori Maehara (maehara@prefield.com)
