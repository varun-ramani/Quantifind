import torch
from torch.utils.data import Dataset
import numpy as np

from workflow import data_workflow

class SimpleDenseDataset(Dataset):
    def __init__(self):
        # Set fixed seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate random input vectors (24000, 256)
        self.inputs = torch.randn(48000, 256)
        
        # Create a fixed random linear transformation (256, 256)
        self.transform_matrix = torch.randn(256, 256)
        
        # Compute outputs using the linear transformation
        self.outputs = torch.matmul(self.inputs, self.transform_matrix)
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
    
    def get_transform_matrix(self):
        """Return the ground truth transformation matrix"""
        return self.transform_matrix.clone()

def create_dataloaders(batch_size=32, train_split=0.8, val_split=0.1,
                       pin_memory=True, seed=42):

    dataset = SimpleDenseDataset()
    return data_workflow.create_dataloaders(dataset)