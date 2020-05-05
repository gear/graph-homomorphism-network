# graph homomorphism profile
Proof of concept for graph homomorphism profile as feature.
http://arxiv.org/abs/2005.01214

Note: There is a bug in homlib for atlas[100:].

To run experiments, install required packages, then:

```
python svm.py --dataset MUTAG --hom_type tree --C 10 --gamma 0.1 --num_run 100
```

Experiment sheet:
```
https://docs.google.com/spreadsheets/d/1-1gXlCyXHpO_DJKdi_XJyvwoa_11K962AYhAVdRk8Fw/edit?usp=sharing
```
