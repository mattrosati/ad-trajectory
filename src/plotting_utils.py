from typing import Dict, Union, Any

from transformers import Trainer
import torch
from torch.utils.data import default_collate
from transformers import TrainerCallback, TrainerState, TrainerControl
import wandb
import matplotlib.pyplot as plt
    
    
def plot_predicted_trajectory(t_query, y_mean, y_std, title="GP-VAE Trajectory", color="C0"):
    """
    Plot GP-VAE prediction with ±1 std shaded.

    Args:
        t_query: [T], timepoints
        y_mean: [T], predicted mean
        y_std: [T], predicted std
    """
    t_query = t_query.detach().cpu().numpy()
    y_mean = y_mean.detach().cpu().numpy()
    y_std = y_std.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_query, y_mean, label="Predicted", color=color)
    ax.fill_between(t_query, y_mean - y_std, y_mean + y_std, alpha=0.3, color=color, label="±1 std")
    ax.set_xlabel("Time")
    ax.set_ylabel("Target value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig