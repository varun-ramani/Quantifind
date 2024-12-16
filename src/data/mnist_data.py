from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import ConcatDataset

from workflow import data_workflow

def create_dataloaders(batch_size=32, train_split=0.8, val_split=0.1,
                       pin_memory=True, seed=42):
    
    mnist_train = MNIST('datasets/mnist', download=True, train=True, transform=ToTensor())
    mnist_test = MNIST('datasets/mnist', download=True, train=False, transform=ToTensor())
    mnist = ConcatDataset([mnist_train, mnist_test])

    return data_workflow.create_dataloaders(mnist)