from torch import nn
import torch.nn.functional as F
import torch
import utils

class ConvMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        # First conv block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.pool1 = nn.MaxPool2d(2)  # 28x28 -> 14x14
        
        # Second conv block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.pool2 = nn.MaxPool2d(2)  # 14x14 -> 7x7
        
        # Third conv block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 7x7 -> 7x7
        self.pool3 = nn.MaxPool2d(2)  # 7x7 -> 3x3 (with rounding down)
        
        # Final dense layer
        self.dense = nn.Linear(128 * 3 * 3, 10)

    def forward(self, x: torch.Tensor):
        # Conv blocks with ReLU activation
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten and dense layer
        x = x.flatten(1)
        x = self.dense(x)
        
        # Softmax activation
        x = F.softmax(x, dim=-1)
        
        return x

def create_model_context():
    net = ConvMNISTModel().to(utils.torch_device)
    crit = nn.CrossEntropyLoss().to(utils.torch_device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    return net, crit, optimizer