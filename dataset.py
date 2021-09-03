import config

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from sklearn.datasets import make_classification, make_regression




np.random.seed(42)

x_reg, y_reg = make_regression(n_samples=1000, n_features=20, shuffle=True, n_targets=1)


class CustomDataset(Dataset):

    def __init__(self, x_reg, y_reg):
        self.x = torch.as_tensor(x_reg).float()
        self.y = torch.as_tensor(y_reg).float()

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


dataset = CustomDataset(x_reg, y_reg)


ratio = .8
n_total = len(dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_train

# split the training and validation data
train_data, val_data = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(
    dataset= train_data,
    batch_size= config.BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    dataset = val_data,
    batch_size = config.BATCH_SIZE
)

