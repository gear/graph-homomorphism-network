# graph homomorphism convolution

## Required packages and datasets

### Packages
```
numpy
scikit-learn
homlib (for non-tree homomorphism)
hyperopt-sklearn (optional, accuracy improves if used)
```

### Datasets
Data is included in the `data/` folder. These datasets are downloaded from the
TU Dortmund collection using `torch-geometric` data utils. We packed the data
using `packer.py` so user don't have to install `torch-geometric`.

For synthetic datasets, we provide the dataloader for GIN and GNTK in
`gin_utils.py` and `gntk_utils.py`. Note that we force 'use degree as tag'
by default.

## Installation 

1. Install `homlib` with the included instructions in `homlib.zip`. We added
a note for Anaconda compatibility.
2. Install other required packages with `pip` or `conda`.

## Experiments

### Benchmark datasets
To run experiments with the included datasets (TUD datasets). Check `tud.py`
for more commandline options. 

Example:
```
python tud.py --dataset [MUTAG,IMDB-BINARY,IMDB-MULTI]
              --hom_type [tree,cycle,label_tree]
              --hom_size [6,8]
```

### Synthetic datasets
Example:
```
python synthetic.py --dataset [bipartite,paulus25,csl]
              --hom_type [tree,cycle]
              --hom_size [6,8]
```

### Other methods
Check `externals.py` and use `load_packed_tud` function to load CSL and PAULUS.
Use `gen_bipartite` to generate bipartite graphs.
