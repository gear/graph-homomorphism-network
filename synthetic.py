import argparse
import numpy as np
from tqdm import tqdm
from time import time
from utils import load_data, load_precompute, save_precompute,\
                  load_tud_data, load_packed_tud
from utils import get_scaler
from utils import gen_bipartite
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from homomorphism import get_hom_profile


parser = argparse.ArgumentParser('Synthetic data experiments.')
# Data loader
parser.add_argument("--dataset", type=str, help="Synthetic dataset name to run.")
parser.add_argument("--ngraphs", type=int, help="Number of graph per class.",
                    default=200)
# Parameters for homomorphism
parser.add_argument("--hom_type", type=str, help="Type of homomorphism.")
parser.add_argument("--hom_size", type=int, default=6,
                    help="Max size of F graph.")
parser.add_argument("--hom_density", action="store_true", default=False,
                    help="Compute homomorphism density instead of count.")
# Hyperparams for SVM
parser.add_argument("--C", type=float, help="SVC's C parameter.", default=1e4)
parser.add_argument("--kernel", type=str, help="SVC kernel function.", 
                    default="rbf")
parser.add_argument("--degree", type=int, help="Degree of `poly` kernel.",
                    default=2)
parser.add_argument("--gamma", type=float, help="SVC's gamma parameter.",
                    default=40.0)
# Misc
parser.add_argument("--num_run", type=int, default=10,
                    help="Number of experiments to run.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--grid_search", action="store_true", default=False)
parser.add_argument("--gs_nfolds", type=int, default=5)
parser.add_argument("--disable_hom", action="store_true", default=False)
parser.add_argument("--f1avg", type=str, default="micro",
                    help="Average method for f1.")
parser.add_argument("--scaler", type=str, default="standard",
                    help="Name of data scaler to use as the preprocessing step")


# Default grid for SVC
Cs = np.logspace(-5, 6, 120)
gammas = np.logspace(-5, 1, 20)
class_weight = ['balanced']
param_grid = {'C': Cs, 'gamma': gammas, 'class_weight': class_weight}


if __name__ == "__main__":
    ### Load data and computer homomorphism 
    args = parser.parse_args()
    hom_time = 0
    learn_time = 0
    # Choose function to load data
    if args.dataset == "bipartite":
        data, nclass = gen_bipartite(args.ngraphs)
    y = [d.label for d in data]
    y = np.array(y)
    node_features = None
    X = []
    hom_time = time()
    profile_func = get_hom_profile(args.hom_type)
    print("Computing {} homomorphism...".format(args.hom_type))
    for d in tqdm(data):
        profile = profile_func(d.g, size=args.hom_size, 
                               density=args.hom_density,
                               node_tags=None)
        X.append(profile)
        hom_time = time() - hom_time
    X = np.array(X, dtype=float)

    ### Train SVC 
    print("Training SVM...")
    learn_time = time()
    a_acc = []  # All accuracies of num_run
    a_std = []  # All standard deviation
    for j in tqdm(range(args.num_run)):
        acc = []
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        scaler = get_scaler(args.scaler)
        X = scaler.fit_transform(X)
        grid_search = GridSearchCV(SVC(kernel=args.kernel), param_grid, 
                                           iid=False, cv=skf, 
                                           n_jobs=8)
        grid_search.fit(X,y)
        idx = grid_search.best_index_
        a_acc.append(grid_search.cv_results_['mean_test_score'][idx])
        a_std.append(grid_search.cv_results_['std_test_score'][idx])
    learn_time = time() - learn_time

    print("Accuracy: {:.4f} +/- {:.4f}".format(np.mean(a_acc), np.mean(a_std)))
    print("Time for homomorphism: {:.2f} sec".format(hom_time))
    print("Time for SVM: {:.2f} sec".format(learn_time))
