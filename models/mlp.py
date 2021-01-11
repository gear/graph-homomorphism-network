import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

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
        return x


if __name__ == "__main__":
    import os
    import uuid
    import tqdm
    import argparse
    import pickle as pkl
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from ghc.homomorphism import get_hom_profile
    from ghc.data_utils import load_data, load_precompute, save_precompute

    hom_types = get_hom_profile(None)

    #### Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='MUTAG')
    parser.add_argument('--hom_type', type=str, choices=hom_types)
    parser.add_argument('--home_size', type=int, default=6)
    parser.add_argument('--dloc', type=str, default="./data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=0.00005)
    parser.add_argument('--hids', type=int, nargs='+', default=[64])
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--cuda', action="store_true", default=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument("--log_period", type=int, default=50)
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
    checkpt_file = 'checkpoints/'+uuid.uuid4().hex[:4]+'-'+args.data+'.pt'
    print(device_id, checkpt_file)
    #### Load data and compute homomorphism
    graphs, X, y = load_data(args.data.upper(), args.dloc)
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
    tensory = torch.Tensor(y).float().to(device)
    #### Initialize learning model
    model = MLP(tensorX.size(-1), tensory.max()+1, args.hids,
                dp=args.dropout).to(device)
    opt_config = {'params': model.parameters(),
                  'weight_decay': args.wd,
                  'lr': args.lr}
    optimizer = optim.Adam(opt_config)
    #### Train and Test functions
    def train(m, o, idx_train, idx_test):
        idx_train = torch.Tensor(idx_train).to(device)
        idx_test = torch.Tensor(idx_test).to(device)
        m.train()
        o.zero_grad()
        output = model(tensorX)
        acc_train = accuracy(output[idx_train], tensory[idx_train])
        loss_train = F.nll_loss(output[idx_train], tensory[idx_train])
        loss_train.backward()
        optimizer.step()
        return loss_train.item(), acc_train.item()
















