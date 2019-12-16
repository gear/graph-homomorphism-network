import argparse
import numpy as np
from tqdm import tqdm
from time import time
from utils import load_data, load_precompute, save_precompute, load_tud_data
from utils import get_scaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from homomorphism import get_hom_profile

TUD_datasets = {
    "COX2",
    "DD",
    "ENZYMES",
    "COX2_MD",
    "NCI109",
    "BZR",
    "BZR_MD"
}

parser = argparse.ArgumentParser('SVM with homomorphism profile.')
# Data loader
parser.add_argument("--dataset", type=str, help="Dataset name to run.")
parser.add_argument("--test_ratio", type=float, help="Test split.", default=0.1)
parser.add_argument("--precompute", action="store_true", default=False, 
                    help="Precomputed homomorphism count.")
parser.add_argument("--feature", type=str, default="skip",
                    help="How to handle node feature. [skip or append].")
parser.add_argument("--combine_feature_tag", action="store_true", default=False,
                    help="Append features and tags in TUD datasets.")
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
parser.add_argument("--disable_hom", action="store_true", default=False)
parser.add_argument("--scaler", type=str, default="standard",
                    help="Name of data scaler to use as the preprocessing step")


# Default grid for SVC
Cs = np.logspace(-3, 5, 250)
gammas = ['scale']
class_weight = ['balanced']
param_grid = {'C': Cs, 'gamma': gammas, 'class_weight': class_weight}


if __name__ == "__main__":
    args = parser.parse_args()
    hom_time = 0
    svm_time = 0
    # Choose function to load data
    if args.dataset in TUD_datasets:
        load_data = load_tud_data
    else:
        args.combine_feature_tag = False
    # If use labeled homomorphism, node tags of GIN loader must be one hot. 
    if args.hom_type.find("labeled") > -1:
        data, nclass = load_data(args.dataset, args.combine_feature_tag,
                                 onehot_tags=True)
    else:
        data, nclass = load_data(args.dataset, args.combine_feature_tag)
    X = []
    y = [d.label for d in data]
    y = np.array(y)
    node_features = None
    compute_X = False
    if data[0].node_features is not None:
        # Monkey patch for compatibility with GIN's dataloader
        try:
            node_features = [d.node_features.sum(0).numpy() for d in data]
        except AttributeError:
            node_features = [d.node_features.sum(0) for d in data]
        node_features = np.array(node_features)
        dim_features = node_features.shape[1]
    # Compute (single type) homomorphism profile
    if args.precompute:
        X = load_precompute(args.dataset, args.hom_type, args.hom_size)
    # If X is []
    if len(X) == 0:
        compute_X = True
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
    # Save homomorphism profile if precompute flag is set and X is computed
    if args.precompute and compute_X:
        save_precompute(X, args.dataset, args.hom_type, args.hom_size)
    dim_hom = X.shape[1]
    if args.disable_hom:
        X = np.zeros_like(X)
    if args.feature == "append" and node_features is not None:
        print("Appending features...")
        X = np.concatenate((X, node_features), axis=1)
    # Train SVC 
    print("Training SVM...")
    svm_time = time()
    a_acc = []  # All accuracies of num_run
    a_std = []
    for j in tqdm(range(args.num_run)):
        skf = StratifiedKFold(n_splits=int(1/args.test_ratio), shuffle=True)
        # Fit a scaler to training data
        scaler = get_scaler(args.scaler)
        scaler = scaler.fit(X)
        X = scaler.transform(X)
        clf = GridSearchCV(SVC(), param_grid, 
                           iid=False, cv=skf, 
                           n_jobs=8)
        clf.fit(X, y)
        bid = clf.best_index_
        a_acc.append(clf.best_score_)
        a_std.append(clf.cv_results_['std_test_score'][bid])
    svm_time = time() - svm_time
    print("Accuracy: {:.4f} +/- {:.4f}".format(np.mean(a_acc), np.mean(a_std)))
    print("Time for homomorphism: {:.2f} sec".format(hom_time))
    print("Time for SVM: {:.2f} sec".format(svm_time/(args.num_run*\
                                                      int(1/args.test_ratio))))
