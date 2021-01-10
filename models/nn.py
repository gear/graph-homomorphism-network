import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hiddens, **kwargs):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fcs = []
        prev_dim = in_dim
        for h in hiddens:
            self.fcs.append(nn.Linear(prev_dim, h))
            prev_dim = h
        self.fcs.append(nn.Linear(hiddens[-1], out_dim))
        self.fcs = nn.ModuleList(self.fcs)
        self.normalizations = nn.ModuleList([
            nn.LayerNorm(in_dim),
            *[nn.LayerNorm(h) for h in hiddens]
        ])
        #### Extra arguments, e.g. dropout (dp)  
        if 'dp' in kwargs:
            self.dp = nn.Dropout(p=kwargs['dp'], inplace=True)
        else:
            self.dp = None

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = self.normalizations[i](x)
            x = fc(x)
            x = F.relu(x)
            if self.dp is not None and i < len(self.fcs)-1:
                x = self.dp(x)
        return x
