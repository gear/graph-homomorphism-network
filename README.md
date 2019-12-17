# graph homomorphism profile
Proof of concept for graph homomorphism profile as feature.
@maehara-san: The name of this repo is due to the simple proof paper, when we called it "network".

To run experiments, install required packages, then:

```
python svm.py --dataset MUTAG --hom_type tree --C 10 --gamma 0.1 --num_run 100
```

Experiment sheet:
```
https://docs.google.com/spreadsheets/d/1-1gXlCyXHpO_DJKdi_XJyvwoa_11K962AYhAVdRk8Fw/edit?usp=sharing
```

To run hyperopt:
cd to the folder
pip install -e .
