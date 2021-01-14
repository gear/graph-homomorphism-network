import os
import csv
import uuid
import argparse
import pickle as pkl
from time import time
from itertools import product
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from ghc.homomorphism import get_hom_profile
from ghc.utils.data import load_data, load_precompute,\
                               save_precompute, load_folds
from ghc.utils.ml import accuracy
import sys
from mlp import MLP


hom_types = get_hom_profile(None)

params_grid = {
    'lr': [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
    'wd': [0.000005, 0.00005, 0.0005, 0.005, 0.05, 0.1],
    'dropout': [0.1, 0.3, 0.5, 0.7, 0.8],
    'patience': [20, 100, 200, 400, 800]
}

#### Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='MUTAG')
parser.add_argument('--hom_type', type=str, choices=hom_types)
parser.add_argument('--hom_size', type=int, default=6)
parser.add_argument('--dloc', type=str, default="./data")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--hids', type=int, nargs='+', default=[64, 64, 64])
parser.add_argument('--cuda', action="store_true", default=False)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument("--log_period", type=int, default=500)
args = parser.parse_args()
#### Setup devices and random seeds
torch.manual_seed(args.seed)
device_id = "cpu"
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device_id = "cuda:"+str(args.gpu_id)
device = torch.device(device_id)
#### Setup checkpoints
os.makedirs("./checkpoints/", exist_ok=True)
os.makedirs(os.path.join(args.dloc, "precompute"), exist_ok=True)
#### Load data and compute homomorphism
graphs, X, y = load_data(args.data.upper(), args.dloc)
splits = load_folds(args.data.upper(), args.dloc)
hom_func = get_hom_profile(args.hom_type)
try:
    homX = load_precompute(args.data.upper(),
                    args.hom_type,
                    args.hom_size,
                    os.path.join(args.dloc, "precompute"))
except FileNotFoundError:
    homX = [hom_func(g, size=args.hom_size, density=False, node_tags=X)\
            for g in tqdm(graphs, desc="Hom")]
    save_precompute(homX, args.data.upper(), args.hom_type, args.hom_size,
                    os.path.join(args.dloc, "precompute"))
tensorX = torch.Tensor(homX).float().to(device)
tensorX = tensorX / (tensorX.max(0, keepdim=True)[0]+0.5)
tensory = torch.Tensor(y).flatten().long().to(device)
tensorX.requires_grad_(requires_grad=False)
tensory.requires_grad_(requires_grad=False)

#### Train and Test functions
def train(m, o, idx):
    m.train()
    o.zero_grad()
    output = m(tensorX)
    acc_train = accuracy(output[idx], tensory[idx])
    loss_train = F.nll_loss(output[idx], tensory[idx])
    loss_train.backward()
    o.step()
    return loss_train.item(), acc_train.item()

def test(m, idx, checkpt_file):
    m.load_state_dict(torch.load(checkpt_file))
    m.eval()
    with torch.no_grad():
        output = m(tensorX)
        loss_test = F.nll_loss(output[idx], tensory[idx])
        acc_test = accuracy(output[idx], tensory[idx])
        return loss_test.item(), acc_test.item()

#### Run for one config of hyper-params to get a 10-folds scores
def cv_score(lr, wd, dropout, patience, logger):
    scores = []
    for split in splits:
        model = MLP(tensorX.size(-1), int(tensory.max()+1), args.hids,
                    dp=dropout).to(device)
        opt_config = [{'params': model.parameters(),
                       'weight_decay': wd,
                       'lr': lr}]
        optimizer = optim.Adam(opt_config)
        idx_train, idx_test = split
        idx_train = torch.Tensor(idx_train).long().to(device)
        idx_test = torch.Tensor(idx_test).long().to(device)
    
        checkpt_file = 'checkpoints/'+uuid.uuid4().hex[:4]+'-'+args.data+'.pt'
        print(device_id, checkpt_file)
        c = 0
        best = 0
        best_epoch = 0
        acc = 0
        for epoch in range(args.epochs):
            loss_train, acc_train = train(model, optimizer, idx_train)
            if args.log_period!=-1 and ((epoch+1)%args.log_period == 0 or epoch == 0):
                print('Epoch:{:04d}'.format(epoch+1),
                    'loss:{:.3f}'.format(loss_train),
                    'acc:{:.2f}'.format(acc_train*100))
            if acc_train > best:
                best = acc_train
                best_epoch = epoch
                torch.save(model.state_dict(), checkpt_file)
                c = 0
            else:
                c += 1
            if c == patience:
                break
        _, test_acc = test(model, idx_test, checkpt_file)
        scores.append(test_acc)
    avg = np.mean(scores)
    std = np.std(scores)
    with open(logger, 'a', newline='') as log:
        log_writer = csv.writer(log, delimiter=',', quotechar='|')
        log_writer.writerow([lr, wd, dropout, patience, avg, std])
    return avg, std

for lr, wd, dropout, patience in product(*params_grid.values()):
    print(cv_score(lr, wd, dropout, patience, 
          "{}.csv".format(args.data.upper())))
