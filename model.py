import config

import torch
import torch.nn as nn


class LinearRegression(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(config.INPUT_FEATURES, config.HIDDEN_SIZE)
        self.layer_2 = nn.Linear(config.HIDDEN_SIZE, config.OUTPUT_FEATURES)
        self.relu = nn.ReLU()
    

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        out = self.layer_2(x)
        return out



