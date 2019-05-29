import torch
import os
import torch.utils.data as data
import numpy as np
import pandas as pd

class movielensDataset(data.Dataset):
    def __init__(self, train=True):
        self.train = train
        if self.train == True:
            self.ua = pd.read_csv('../input/ml-100k/ua.base', delimiter='\t', iterator=False)
        else:
            self.ua = pd.read_csv('../input/ml-100k/ua.test', delimiter='\t', iterator=False)

    def __len__(self):
        if self.train == True:
            return 90569
        else:
            return 9429

    def __getitem__(self, index):
        raw_data = self.ua.values
        data = raw_data[index,0:2]
        label = raw_data[index,2]
        return data, label