from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import ConcatDataset, Dataset
import torch

from workflow import data_workflow
from utils import torch_device

class MNISTDataset(Dataset):
    def __init__(self):
        mnist_train = MNIST('datasets/mnist', download=True, train=True, transform=ToTensor())
        mnist_test = MNIST('datasets/mnist', download=True, train=False, transform=ToTensor())
        mnist = ConcatDataset([mnist_train, mnist_test])
        self.items = [
            (x.to(torch_device), y)
            for (x, y) 
            in mnist
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, x):
        return self.items[x]

def create_dataloaders(batch_size=32, train_split=0.8, val_split=0.1,
                       pin_memory=True, seed=42):
    
    mnist = MNISTDataset()
    return data_workflow.create_dataloaders(mnist)