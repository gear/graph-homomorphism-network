# Graph Homomorphism Convolution
Proof of concept for Graph Homomorphism Convolution.
http://arxiv.org/abs/2005.01214 (ICML'20)

Note: Code for left homomorphism is for our ICML'20 paper.
Code for right homomorphism is our continued work.

## Setup
To obtain required packages and install our modules, run
```
pip install -r requirements.txt
```
This script installs `homlib` and `ghc`.

## Download and pack data
This package is not dependent on `torch-geometric` but we will use 
`torch-geometric` to load and pack data. Download `data.zip` from
```
https://drive.google.com/file/d/15w7UyqG_MjCqdRL2fA87m7-vanjddKNh/view?usp=sharing
```
and extract to `data/`. For example MUTAG dataset should have: `data/MUTAG.graph`, `data/MUTAG.X`, and `data/MUTAG.y`. 

## Run experiments
Experiment scripts are placed in the top level of this repository and named 
by the machine learning model. In general, a 10-fold CV score is reported.
For example,
```
python svm.py --data mutag --hom tree --size 8 
python svm.py --data mutag --hom feature_tree --size 8 
```

Cite us as:
```
@InProceedings{
    nt20ghc, 
    title = {Graph Homomorphism Convolution}, 
    author = {NT, Hoang and Maehara, Takanori}, 
    booktitle = {Proceedings of the 37th International Conference on Machine Learning}, 
    pages = {7306--7316}, 
    year = {2020}, 
    editor = {Hal Daumé III and Aarti Singh}, 
    volume = {119}, 
    series = {Proceedings of Machine Learning Research}, 
    address = {Virtual}, 
    month = {13--18 Jul}, 
    publisher = {PMLR}, 
    pdf = {http://proceedings.mlr.press/v119/nguyen20c/nguyen20c.pdf},, 
    url = {http://proceedings.mlr.press/v119/nguyen20c.html}, 
}
```

Note: There is a bug in homlib for `atlas[100:]`.
