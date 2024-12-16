from torch import nn
import torch

from pathlib import Path

class SimpleDenseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(256, 256)

    def forward(self, x):
        return nn.Sequential(self.layer, nn.ReLU())(x)

def create_model_context():
    net = SimpleDenseModel()
    crit = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    return net, crit, optimizer