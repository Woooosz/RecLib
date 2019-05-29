import torch
from torch import nn
import numpy as np 
import pandas as pd
from torch import nn

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = nn.Embedding(n_users + 1, n_factors, sparse=True)
        self.item_factors = nn.Embedding(n_items + 1, n_factors, sparse=True)
        
        self.user_bias = nn.Embedding(n_users + 1, 1, sparse=True)
        self.item_bias = nn.Embedding(n_items + 1, 1, sparse=True)

    def forward(self, user, item):
        output = (self.user_bias(user) + self.item_bias(item)).squeeze(1)
        output += (self.user_factors(user) * self.item_factors(item)).sum(1)
        return output
    