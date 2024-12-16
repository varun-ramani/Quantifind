from rich.console import Console
import torch

console = Console()
torch_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def log_error(message, webhook=True):
    console.log(message, style='red')

def log_info(message, webhook=True):
    console.log(message, style='blue')

def log_warning(message, webhook=True):
    console.log(message, style='yellow')

log_info(f'Using torch device {torch_device.type}')