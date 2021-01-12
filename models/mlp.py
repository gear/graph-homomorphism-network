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
from tqdm import tqdm
from ghc.homomorphism import get_hom_profile
from ghc.utils.data import load_data, load_precompute,\
                               save_precompute, load_folds
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
        self.normalizations = nn.ModuleList([
            nn.LayerNorm(in_dim),
            *[nn.LayerNorm(h) for h in hiddens]
        ])
        self.dp = nn.Dropout(dp, inplace=True)

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = self.normalizations[i](x)
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
    parser.add_argument('--dloc', type=str, default="./data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0.00005)
    parser.add_argument('--hids', type=int, nargs='+', default=[64, 64])
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=500)
    parser.add_argument('--cuda', action="store_true", default=False)
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
    #### Setup checkpoints
    os.makedirs("./checkpoints/", exist_ok=True)
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
    #### Run for 10-folds scores
    for split in splits:
        model = MLP(tensorX.size(-1), int(tensory.max()+1), args.hids,
                    dp=args.dropout).to(device)
        opt_config = [{'params': model.parameters(),
                       'weight_decay': args.wd,
                       'lr': args.lr}]
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
        print('Load {}th epoch. Test acc:{:.4f}'.format(best_epoch, test_acc))
