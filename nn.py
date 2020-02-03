import argparse
import numpy as np
from tqdm import tqdm
from time import time
from utils import load_data, load_precompute, save_precompute, load_tud_data
from utils import get_scaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
from hom_conv import HNet 
from data import torch_data
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser("Homomorphism Network with tree profile.")
parser.add_argument("--dataset", type=str, help="Dataset name to run.")
# Params for neural nets
parser.add_argument("--max_tree_size", type=int, default=6, 
                    help="Max tree size.")
parser.add_argument("--hdim", type=int, default=16, help="Hidden dim for clf.")
# Params for training
parser.add_argument("--epoch", type=int, default=10, 
                    help="Number of training epochs.")
parser.add_argument("--lr", type=float, default=0.003, help="Learning rate.")
parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay.")
parser.add_argument("--bs", type=int, default=32, help="Batch size.")


def graph_collate(batch):
    graphs = [d[0][0] for d in batch]
    X = [d[0][1] for d in batch]
    y = torch.tensor([d[1] for d in batch]).reshape(-1).long()
    return (graphs, X), y


def train_val(net, tdset, vdset, args):
    train_generator = DataLoader(tdset, batch_size=args.bs, 
                                 collate_fn=graph_collate)
    val_generator = DataLoader(vdset, batch_size=len(vdset), 
                               collate_fn=graph_collate)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_f = nn.CrossEntropyLoss()
    logger = defaultdict(list)

    for epoch in range(args.epoch):
        correct_count = 0
        i = 0
        for (batch_graphs, batch_X), batch_y in train_generator:
            logger["epoch"].append(epoch)
            logger["batch"].append(i)
            net.train()
            optimizer.zero_grad()
            out = net(batch_graphs, batch_X)
            train_loss = loss_f(out, batch_y)
            train_loss.backward()
            optimizer.step()
            correct_count += out.argmax(1).eq(batch_y).double().sum()
            logger["batch_loss"].append(train_loss.item())
            logger["train_accuracy"].append(correct_count.item()/len(tdset))
            i += 1
        with torch.no_grad():
            for (graphs, X), y in val_generator:
                net.eval()
                out = net(graphs, X)
                val_loss = loss_f(out, y)
                corrects = out.argmax(1).eq(y).double().sum()
                logger["val_accuracy"].append(corrects.item()/len(vdset))

    return net, logger

        
if __name__ == "__main__":
    args = parser.parse_args()
    assert args.bs > 1, "Batchnorm doesn't work with this batch size."

    # Dataloader 
    dataset = torch_data(args.dataset, num_folds=10)

    i = 1
    for tdset, vdset in dataset.folds():
        net = HNet(dataset.fdim, dataset.nclass, args.hdim, args.max_tree_size)
        net.weights_init()
        print("Fold {}...".format(i))
        net, logger = train_val(net, tdset, vdset, args)
        i+=1
        print(logger)
