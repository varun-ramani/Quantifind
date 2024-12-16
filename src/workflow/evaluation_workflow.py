import torch
from torch import nn
from torch.utils.data import DataLoader
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List, Tuple
import utils
from workflow.quantize_workflow import BitQuantizer

stored_data = {}
def compute_test_loss(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    taskname: Optional[str] = "Computing test loss...",
    progress: Optional[Progress] = None,
) -> float:
    """
    Compute the test loss for a given model and test loader.

    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        loss_fn: Loss function
        progress: Optional Progress bar for visualization

    Returns:
        float: Average test loss
    """
    global stored_data
    task_key = (model, test_loader, loss_fn)
    if task_key not in stored_data:
        should_close_progress = False
        if progress is None:
            progress = Progress(console=utils.console)
            should_close_progress = True
            progress.start()

        task = progress.add_task(taskname, total=len(test_loader))
        total_loss = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                outputs = model(x)
                batch_loss = loss_fn(outputs, y)
                total_loss += batch_loss.item()
                progress.update(task, advance=1)

        if should_close_progress:
            progress.stop()

        progress.update(task, visible=False)

        stored_data[task_key] = total_loss / len(test_loader)

    return stored_data[task_key]


def evaluate_quantized_model(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    bit_depth: int,
    progress: Optional[Progress] = None,
) -> Dict[str, float]:
    global evaluations

    """
    Evaluate original and quantized models at specified bit depth.
    
    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        loss_fn: Loss function
        bit_depth: Number of bits for quantization
        progress: Optional Progress bar
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    should_close_progress = False
    if progress is None:
        progress = Progress(console=utils.console)
        should_close_progress = True
        progress.start()

    # Create quantizer and quantize model/data
    quantizer = BitQuantizer(bit_depth)
    quantized_model = quantizer.quantized_model(model)
    quantized_loader = quantizer.quantized_dataloader(test_loader)

    # Compute losses
    original_loss = compute_test_loss(
        model,
        test_loader,
        loss_fn,
        taskname=("Compute 32-bit test loss"),
        progress=progress,
    )
    quantized_loss = compute_test_loss(
        quantized_model,
        quantized_loader,
        loss_fn,
        taskname=(f"Compute {bit_depth}-bit test loss"),
        progress=progress,
    )

    if should_close_progress:
        progress.stop()

    return {
        "bit_depth": bit_depth,
        "original_loss": original_loss,
        "quantized_loss": quantized_loss,
        "normalized_loss": quantized_loss / bit_depth,
        "loss_ratio": quantized_loss / original_loss,
    }


def evaluate_bit_depths(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    min_bits: int = 2,
    max_bits: int = 32,
    num_workers: int = 4,
) -> List[Dict[str, float]]:
    
    """
    Evaluate model across multiple bit depths using parallel processing.

    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        loss_fn: Loss function
        min_bits: Minimum bit depth to evaluate
        max_bits: Maximum bit depth to evaluate
        num_workers: Number of parallel workers

    Returns:
        list: List of evaluation results for each bit depth
    """
    progress = Progress(console=utils.console)
    progress.start()

    bit_depths = range(min_bits, max_bits + 1)
    results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for bits in bit_depths:
            future = executor.submit(
                evaluate_quantized_model, model, test_loader, loss_fn, bits, progress
            )
            futures.append(future)

        task = progress.add_task("Evaluating bit depths...", total=len(futures))
        for future in futures:
            results.append(future.result())
            progress.update(task, advance=1)

    progress.stop()
    return sorted(results, key=lambda x: x["bit_depth"])
