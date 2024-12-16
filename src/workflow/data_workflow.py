import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import utils

def create_dataloaders(dataset: Dataset, batch_size=32, train_split=0.8, val_split=0.1,
                       pin_memory=True, seed=42):
    """
    Create train, validation, and test DataLoaders for SimpleDenseDataset.
    
    Args:
        batch_size: Number of samples per batch
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        pin_memory: Whether to pin memory for GPU transfer
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    
    # Set seed for reproducible splits
    torch.manual_seed(seed)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator(device=utils.torch_device).manual_seed(seed)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )
    
    return train_loader, val_loader, test_loader