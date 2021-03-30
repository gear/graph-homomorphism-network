import os
import uuid
import argparse
import pickle as pkl
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from ghc.homomorphism import get_hom_profile
from ghc.utils.data import load_data, load_precompute, save_precompute,\
                           load_folds, augment_data
from ghc.utils.ml import accuracy
import sys

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hiddens, dp=0.7, **kwargs):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fcs = []
        prev_dim = in_dim
        for h in hiddens:
            self.fcs.append(nn.Linear(prev_dim, h))
            prev_dim = h
        if hiddens:
            self.fcs.append(nn.Linear(hiddens[-1], out_dim, bias=False))
        else:
            self.fcs.append(nn.Linear(in_dim, out_dim, bias=False))
        self.fcs = nn.ModuleList(self.fcs)
        self.dp = nn.Dropout(dp, inplace=True)

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if self.dp is not None and i < len(self.fcs)-1:
                x = self.dp(x)
                x = F.relu(x)
        return F.log_softmax(x, dim=-1)


if __name__ == "__main__":

    hom_types = get_hom_profile(None)

    #### Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='MUTAG')
    parser.add_argument('--hom_type', type=str, choices=hom_types)
    parser.add_argument('--hom_size', type=int, default=6)
    parser.add_argument('--drop_nodes', action="store_true", default=False)
    parser.add_argument('--drop_nodes_rate', type=int, default=1)
    parser.add_argument('--gen_per_graph', type=int, default=1)
    parser.add_argument('--dloc', type=str, default="./data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0.00005)
    parser.add_argument('--hids', type=int, nargs='+', default=[64, 64, 64])
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=500)
    parser.add_argument('--cuda', action="store_true", default=False)
    parser.add_argument('--verbose', action="store_true", default=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument("--log_period", type=int, default=200)
    args = parser.parse_args()
    #### Setup devices and random seeds
    torch.manual_seed(args.seed)
    device_id = "cpu"
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        device_id = "cuda:"+str(args.gpu_id)
    device = torch.device(device_id)
    #### Setup checkpoints and precompute
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
        homX = [hom_func(g, size=args.hom_size, density=False, node_tags=X[i])\
                for i, g in enumerate(tqdm(graphs, desc="Hom"))]
        save_precompute(homX, args.data.upper(), args.hom_type, args.hom_size,
                        os.path.join(args.dloc, "precompute"))
    #### If data augmentation is enabled
    if args.drop_nodes:
        gen_graphs, gen_X, gen_y = augment_data(graphs, X, y,
                                                args.gen_per_graph,
                                                rate=args.drop_nodes_rate)
        gen_hom_X = [hom_func(g, size=args.hom_size,
                              density=False,
                              node_tags=gen_X)\
                     for g in tqdm(gen_graphs, desc="Hom (aug)")]
        tensor_gen_X = torch.Tensor(gen_hom_X).float().to(device)
        scaler = (tensor_gen_X.max(0, keepdim=True)[0] + 0.5)
        tensor_gen_X = tensor_gen_X / scaler
        tensor_gen_y = torch.Tensor(gen_y).flatten().long().to(device)
    else:
        tensor_gen_X = None
        tensor_gen_y = None
    tensorX = torch.Tensor(homX).float().to(device)
    tensorX = tensorX / (tensorX.max(0, keepdim=False)[0] + 0.5)
    tensory = torch.Tensor(y).flatten().long().to(device)
    #### Train and Test functions
    def train(m, o, idx, tX, ty):
        m.train()
        o.zero_grad()
        output = m(tX)
        acc_train = accuracy(output[idx], ty[idx])
        loss_train = F.nll_loss(output[idx], ty[idx])
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
    #### Run for 10-folds scores
    scores = []
    for split in tqdm(splits, desc="10-folds"):
        model = MLP(tensorX.size(-1), int(tensory.max()+1), args.hids,
                    dp=args.dropout).to(device)
        opt_config = [{'params': model.parameters(),
                       'weight_decay': args.wd,
                       'lr': args.lr}]
        optimizer = optim.Adam(opt_config)
        idx_train, idx_test = split
        idx_train = torch.Tensor(idx_train).long().to(device)
        idx_test = torch.Tensor(idx_test).long().to(device)
        if args.drop_nodes:
            expander = lambda k: [k*args.gen_per_graph+i for i in\
                                  range(args.gen_per_graph)]
            idx_train_gen = torch.Tensor([i for j in idx_train for\
                                          i in expander(j)])
            idx_train_gen = torch.Tensor(idx_train_gen).long().to(device)
        else:
            idx_train_gen = None
        checkpt_file = 'checkpoints/'+uuid.uuid4().hex[:4]+'-'+args.data+'.pt'
        if args.verbose:
            print(device_id, checkpt_file)
        c = 0
        best = 0
        best_epoch = 0
        acc = 0
        for epoch in range(args.epochs):
            if args.drop_nodes:
                _, _ = train(model, optimizer, idx_train_gen,
                             tensor_gen_X, tensor_gen_y)
            loss_train, acc_train = train(model, optimizer, idx_train,
                                          tensorX, tensory)
            if args.verbose:
                if (epoch+1)%args.log_period == 0 or epoch == 0:
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
            if c == args.patience:
                break
        _, test_acc = test(model, idx_test, checkpt_file)
        scores.append(test_acc)
    scores = np.array(scores)
    print('CV score:{:.4f}, {:.4f}'.format(scores.mean(), scores.std()))
