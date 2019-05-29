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

    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)
    