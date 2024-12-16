import torch

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
