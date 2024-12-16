import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy


class BitQuantizer:
    def __init__(self, num_bits=32):
        # reserve the sign bit
        self.num_bits = num_bits - 1

        self.n_levels = 2**(self.num_bits)
        self.min_level = -self.n_levels
        self.max_level = self.n_levels - 1

    def calculate_scale_zero_point(self, min_val: float, max_val: float):
        """
        Calculate the scale and zero point for quantization.
        """
        max_val = max(abs(max_val), abs(min_val))
        min_val = -max_val
            
        # Calculate scale
        scale = (max_val - min_val) / (self.max_level - self.min_level)
        
        zero_point = 0  # For signed, we typically use 0 as zero point
        
        return scale, int(zero_point)
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize input tensor to specified bit precision.
        """
        min_val, max_val = x.min().item(), x.max().item()
        scale, zero_point = self.calculate_scale_zero_point(min_val, max_val)
        
        # Quantize
        x_int = torch.round(x / scale + zero_point)
        
        # Clamp to ensure values are within range
        x_int = torch.clamp(x_int, self.min_level, self.max_level)
        
        # Dequantize
        x_quant = (x_int - zero_point) * scale
        
        return x_quant, scale, zero_point
    
    def quantized_model(self, model: nn.Module) -> nn.Module:
        """
        Creates a quantized version of the supplied neural network by copying it
        and then quantizing all the parameter sets in the copy.
        """
        # Create a deep copy of the model to avoid modifying the original
        quantized_model = copy.deepcopy(model)
        
        # Quantize all parameters in the model, regardless of requires_grad
        for name, param in quantized_model.named_parameters():
            # Quantize the parameter
            quantized_param, _, _ = self.quantize(param.data)
            # Replace the parameter with its quantized version
            param.data.copy_(quantized_param)
        
        return quantized_model

    class QuantizedDataset(Dataset):
        """Inner dataset class to handle quantization of individual samples"""
        def __init__(self, original_dataset, quantizer):
            self.dataset = original_dataset
            self.quantizer = quantizer
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            data, label = self.dataset[idx]
            if isinstance(data, torch.Tensor):
                quantized_data, _, _ = self.quantizer.quantize(data)
                return quantized_data, label
            else:
                # Handle non-tensor data by converting to tensor first
                data_tensor = torch.tensor(data)
                quantized_data, _, _ = self.quantizer.quantize(data_tensor)
                return quantized_data, label

    def quantized_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """
        Creates a quantized version of the supplied DataLoader by quantizing all
        the inputs and leaving the labels alone. The new dataloader should
        retain the same length as the old dataloader (i.e. it shouldn't just be
        an iterator). However, it should not accomplish this by reading the
        entire old dataloader into memory and quantizing it one element at a time.
        """
        # Create a quantized version of the dataset
        quantized_dataset = self.QuantizedDataset(dataloader.dataset, self)
        
        # Create new dataloader with same parameters as original
        return DataLoader(
            dataset=quantized_dataset,
            batch_size=dataloader.batch_size,
            sampler=dataloader.sampler,
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn,
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last,
            timeout=dataloader.timeout,
            worker_init_fn=dataloader.worker_init_fn
        )