from torch import nn
import torch.nn.functional as F
import torch
import utils

from pathlib import Path

class SimpleDenseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(256, 256)

    def forward(self, x):
        return F.relu(self.layer(x))

def create_model_context():
    net = SimpleDenseModel().to(utils.torch_device)
    crit = nn.MSELoss().to(utils.torch_device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    return net, crit, optimizer