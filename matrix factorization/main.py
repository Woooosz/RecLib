import torch
from data_loader import movielensDataset
from torch.utils.data import DataLoader
from model import MatrixFactorization
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 5
batch_size=100
learning_rate = 0.01
n_users = 943
n_items = 1682
total_step = 800

if __name__ == '__main__':
    trdata = movielensDataset(train=True)
    train_loader = DataLoader(trdata, batch_size=128, shuffle=True)
    model = MatrixFactorization(n_users=n_users, n_items = n_items).to(device)
    weight_p, bias_p = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SparseAdam(
        [{'params':weight_p, 'weight_decay':1e-5},
        {'params':bias_p, 'weight_decay':0}
        ], lr=learning_rate)
    for epoch in range(num_epochs):
        for i, (data, label) in enumerate(train_loader):
            label = label.float().to(device) # 转换成tensor float
            user, item = data[:,0].to(device), data[:,1].to(device)
            # Forward pass
            outputs = model(user, item)
            loss = criterion(outputs, label)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Test model
    test_data = movielensDataset(train=False)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True)
    with torch.no_grad():
        total_loss = 0
        total = 0
        for data, label in test_loader:
            label = label.float().to(device)
            user, item = data[:,0].to(device), data[:,1].to(device)
            outputs = model(user, item)
            loss = criterion(outputs, label)
            total += 1
            total_loss += loss
        print('MSE of loss on the 9430 test sample: {}'.format(total_loss / total))
        