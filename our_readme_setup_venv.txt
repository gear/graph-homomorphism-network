##pre-setup
install python3.7 (3.8 appears to fail)
get data from https://drive.google.com/file/d/15w7UyqG_MjCqdRL2fA87m7-vanjddKNh/view?usp=sharing 
extract data to graph-homomorphism-network/data

## setup
git clone https://github.com/gear/graph-homomorphism-network.git
cd graph-homomorphism-network
python3.7 -m venv graph_homs_learning3-7
source ./graph_homs_learning3-7/bin/activate
pip install -r requirements_dev.txt

## test run
python test_run.py
% or
python models/mlp.py --data mutag --hom_type tree --hom_size 6 
