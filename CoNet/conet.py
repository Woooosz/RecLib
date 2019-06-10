import torch
from torch import nn
import numpy as np 
import pandas as pd
from torch import nn
from torch.nn import functional as F

class CoNet(nn.Module):
    def __init__(self, n_users, n_source_item, n_target_item, n_factors=80):
        super().__init__()
        self.user_factors = nn.Embedding(n_users + 1, n_factors, sparse=True)
        self.source_item_factors = nn.Embedding(n_source_item + 1, n_factors, sparse=True) #(1, 160)
        self.target_item_factors = nn.Embedding(n_target_item + 1, n_factors, sparse=True) #(1, 160)


        self.source_fc1 = nn.Linear(n_factors * 2, 128) #(160, 128)
        self.target_fc1 = nn.Linear(n_factors * 2, 64) #(160,64)
        self.cross_v1 = nn.Parameter(torch.randn(64, 128)) # (64,128)


        self.source_fc2 = nn.Linear(128, 64)
        self.target_fc2 = nn.Linear(64, 32)
        self.cross_v2 = nn.Parameter(torch.randn(32, 64))

        self.source_fc3 = nn.Linear(64, 32)
        self.target_fc3 = nn.Linear(32, 16)

        self.source_out = nn.Linear(32, 2)
        self.target_out = nn.Linear(16, 2)

    def forward(self, user, source_item, target_item):
        source_embedding = torch.cat((self.user_factors(user), self.source_item_factors(source_item)), 1) # (1,160)
        target_embedding = torch.cat((self.user_factors(user), self.target_item_factors(target_item)) ,1) # (1,160)
        out_source_v1 = F.relu(self.source_fc1(source_embedding) + torch.mm(self.target_fc1(target_embedding), self.cross_v1)) # (1 * 128) + (1 * 64) * (64 * 128)
        out_target_v1 = F.relu(self.target_fc1(target_embedding) + torch.mm(self.source_fc1(source_embedding), torch.t(self.cross_v1))) #(1 * 64) +(1,128) (128, 64)

        out_source_v2 = F.relu(self.source_fc2(out_source_v1) +  torch.mm(self.target_fc2(out_target_v1), self.cross_v2)) #(1,64) + (1, 32) * (32, 64)
        out_target_v2 = F.relu(self.target_fc2(out_target_v1) + torch.mm(self.source_fc2(out_source_v1), torch.t(self.cross_v2))) # (1,32) + (1, 64) * (64, 32)
        out_source = F.relu(self.source_fc3(out_source_v2))
        out_target = F.relu(self.target_fc3(out_target_v2))
        out_source = self.source_out(out_source)
        out_target = self.target_out(out_target)
        return out_source, out_target
