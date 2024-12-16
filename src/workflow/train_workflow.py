import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from collections import defaultdict
import datetime
from pathlib import Path

from rich.progress import Progress, MofNCompleteColumn
from rich.live import Live
from rich.table import Table

import utils



def train_network(network: nn.Module, loss: nn.Module, optimizer: torch.optim.Optimizer, train_loader: DataLoader, val_loader: DataLoader, epochs=4):
    train_history = defaultdict(dict)
    global_progress = Progress(
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        "• Validation Loss: {task.fields[val_loss]:.4f}"
    )
    local_progress = Progress(
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        "• Sample Loss: {task.fields[sample_loss]:.4f}"
    )
    table = Table(show_header=False, show_lines=False)
    table.add_row(global_progress)
    table.add_row(local_progress)

    with Live(table, console=utils.console) as live:
        training_task = global_progress.add_task(f"Training...", total=epochs)

        def update_val_loss(epoch):
            val_loss_task = local_progress.add_task('Compute val loss', total=len(val_loader), sample_loss=0)
            acc_sum = 0
            for x, y in val_loader:
                subj_loss = loss(network(x).detach(), y).detach()
                acc_sum += subj_loss
                local_progress.update(val_loss_task, advance=1, sample_loss=subj_loss)

            train_history[f'Epoch {epoch}']['val_loss'] = acc_sum / len(val_loader)
            local_progress.update(val_loss_task, visible=False)
            global_progress.update(training_task, val_loss=acc_sum / len(val_loader))

            utils.log_info(f'Finished epoch [{epoch} / {epochs}] with val loss {acc_sum / len(val_loader)}')
            
        update_val_loss(-1)

        for epoch in range(epochs):
            epoch_task = local_progress.add_task(f"Epoch [{epoch} / {epochs}]", total=len(train_loader), sample_loss=0)
            for x, y in train_loader:
                optimizer.zero_grad()
                subj_loss = loss(network(x), y)
                subj_loss.backward()
                optimizer.step()
                local_progress.update(epoch_task, advance=1, sample_loss=subj_loss)
            local_progress.update(epoch_task, visible=False)
            global_progress.update(training_task, advance=1)
            update_val_loss(epoch)

        global_progress.update(training_task, visible=False)

    return train_history


def save_train_context(
    checkpoints_dir: str, 
    net: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    loss: torch.nn.Module
):
    """
    Writes the current network, optimizer, and loss to a timestamped checkpoint.

    - checkpoints_dir: directory to write to
    - net: model
    - optimizer: current optimizer
    - loss: current loss
    """
    checkpoints_dir = Path(checkpoints_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_filename = f'checkpoint_step1_{timestamp}.pth'
    checkpoint_path = checkpoints_dir / checkpoint_filename
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    utils.log_info(f"Saved '{checkpoint_path}' with loss {loss}")