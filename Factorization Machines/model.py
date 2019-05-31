import torch
from torch import nn
import numpy as np 
import pandas as pd
from torch import nn

class FactorizationMachines(nn.Module):
    def __init__(self, n, n_factors=20):
        super(FactorizationMachines, self).__init__()
        self.linear = nn.Linear(n, 1)
        self.n = n
        self.V = nn.Parameter(torch.randn(n, n_factors))

    def forward(self, x):
        linear = self.linear(x)
        interaction = 0.5 * ( torch.mm(x, self.V).pow(2).sum(1, keepdim=True)  - torch.mm(x.pow(2), self.V.pow(2)).sum(1, keepdim=True))
        return linear + interaction