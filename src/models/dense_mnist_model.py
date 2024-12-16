from torch import nn
import torch.nn.functional as F
import torch
import utils

class DenseMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(28 * 28, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor):
        x = x.flatten(1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.softmax(self.dense3(x), dim=-1)

        return x

def create_model_context():
    net = DenseMNISTModel().to(utils.torch_device)
    crit = nn.CrossEntropyLoss().to(utils.torch_device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    return net, crit, optimizer