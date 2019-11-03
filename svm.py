import argparse
import numpy as np
from tqdm import tqdm
from time import time
from utils import load_data, load_precompute
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from homomorphism import get_hom_profile

parser = argparse.ArgumentParser('SVM with homomorphism profile.')
# Data loader
parser.add_argument("--dataset", type=str, help="Dataset name to run.")
parser.add_argument("--hom_type", type=str, help="Type of homomorphism.")
parser.add_argument("--hom_size", type=int, default=6,
                    help="Max size of F graph.")
parser.add_argument("--test_ratio", type=float, help="Test split.", default=0.1)
parser.add_argument("--precomputed", type=str, 
                    help="Precomputed homomorphism count.", default=None)
parser.add_argument("--feature", type=str, default="skip",
                    help="How to handle node feature. [skip or append].")
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

if __name__ == "__main__":
    args = parser.parse_args()
    hom_time = 0
    svm_time = 0
    f1_mic = []
    f1_mac = []
    accuracy = [] 
    # Load data
    data, nclass = load_data(args.dataset, False)
    X = []
    y = []
    # Compute (single type) homomorphism profile
    if args.precomputed is not None:
        X = load_precompute(args.precomputed)
    else:
        hom_time = time()
        profile_func = get_hom_profile(args.hom_type)
        print("Computing {} homomorphism...".format(args.hom_type))
        for d in tqdm(data):
            profile = profile_func(d.g, size=args.hom_size)
            X.append(profile)
            y.append(d.label)
        hom_time = time() - hom_time
    # Train SVC 
    print("Training SVM...")
    svm_time = time()
    for i in tqdm(range(args.num_run)): 
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, 
                             test_size=args.test_ratio,
                             random_state=np.random.randint(69,6969))
        # Fit a scaler to training data
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        clf = SVC(C=args.C, kernel=args.kernel, degree=args.degree, 
                  gamma=args.gamma, decision_function_shape='ovr')
        clf.fit(X_train, y_train)
        f1_micro = f1_score(y_pred=clf.predict(X_test), y_true=y_test, 
                            average="micro")
        f1_macro = f1_score(y_pred=clf.predict(X_test), y_true=y_test, 
                            average="macro")
        acc = accuracy_score(y_pred=clf.predict(X_test), y_true=y_test)
        f1_mic.append(f1_micro)
        f1_mac.append(f1_macro)
        accuracy.append(acc)
    svm_time = time() - svm_time
    print("Final result for {}:".format(args.dataset))
    print("F1 Micro: {:0.4f} - {:0.4f}".format(np.mean(f1_mic), np.std(f1_mic)))
    print("F1 Macro: {:0.4f} - {:0.4f}".format(np.mean(f1_mac), np.std(f1_mac)))
    print("Accuracy: {:0.4f} - {:0.4f}".format(np.mean(accuracy), 
                                               np.std(accuracy)))
    print("Time for homomorphism: {:.2f} sec".format(hom_time))
    print("Time for SVM: {:.2f} sec".format(svm_time/args.num_run))
