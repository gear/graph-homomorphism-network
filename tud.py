import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import numpy as np
from tqdm import tqdm
from time import time
from utils import load_data, get_scaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
from homomorphism import get_hom_profile
from sklearn.svm import SVC


parser = argparse.ArgumentParser('TUD Datasets with GHC.')
# Data loader
parser.add_argument("--dataset", type=str, help="Dataset name to run.")
parser.add_argument("--scaler", type=str, default="standard")
# Parameters for homomorphism
parser.add_argument("--hom_type", type=str, help="Type of homomorphism.")
parser.add_argument("--hom_size", type=int, default=6,
                    help="Max size of F graph.")
parser.add_argument("--hom_density", action="store_true", default=False,
                    help="Compute homomorphism density instead of count.")

Cs = np.logspace(-2, 5, 20)
gammas = ['scale']
param_grid = {'C': Cs, 'gamma': gammas}

if __name__ == "__main__":
    args = parser.parse_args()
    hom_time = 0
    clf_time = 0
    data, nclass = load_data(args.dataset, False)
    X = []
    y = [d.label for d in data]
    y = np.array(y)
    hom_time = time()
    profile_func = get_hom_profile(args.hom_type)
    print("Computing {} homomorphism...".format(args.hom_type))
    for d in tqdm(data):
        profile = profile_func(d.g, size=args.hom_size, 
                               density=args.hom_density,
                               node_tags=d.node_tags)
        X.append(profile)
    hom_time = time() - hom_time
    X = np.array(X, dtype=float)
    clf_time = time()
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    scaler = get_scaler(args.scaler).fit(X)
    X = scaler.transform(X)
    clf = GridSearchCV(SVC(), param_grid, iid=False, cv=skf, n_jobs=8)
    clf.fit(X,y)
    bid = clf.best_index_
    print(clf.best_score_)
    print(clf.cv_results_['std_test_score'][bid])
    clf_time = time() - clf_time
    print("Time for homomorphism: {:.2f} sec".format(hom_time))
    print("Time for SVM: {:.2f} sec".format(clf_time))
