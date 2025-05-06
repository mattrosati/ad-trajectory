from typing import Dict, Union, Any

from transformers import Trainer
import torch

from torch.utils.data import default_collate
from transformers import TrainerCallback, TrainerState, TrainerControl
import wandb
import matplotlib.pyplot as plt
from plotting_utils import plot_predicted_trajectory
import torch.nn as nn

class GPVAEWandbCallback(TrainerCallback):
    def __init__(self, x_cond, t_cond, t_query, target_idx, log_every_n_steps=10):
        self.x_cond = x_cond
        self.t_cond = t_cond
        self.t_query = t_query
        self.target_idx = target_idx
        self.log_every_n_steps = log_every_n_steps

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None or state.global_step % self.log_every_n_steps != 0:
            return

        wandb.log(self._extract_and_prefix(logs, prefix="train"))

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        model = kwargs.get("model")
        if metrics is not None:
            wandb.log(self._extract_and_prefix(metrics, prefix="eval"))


        # Predict trajectory
        y_mean, y_std = model.predict_target_trajectory(
            x_cond=self.x_cond,
            t_query=self.t_query,
            target_idx=self.target_idx,
            t_cond=self.t_cond
        )

        # Create and log plot
        fig = plot_predicted_trajectory(self.t_query, y_mean, y_std,
                                              title="GP-VAE Predicted Trajectory")
        wandb.log({"eval/trajectory_plot": wandb.Image(fig)})

    def _extract_and_prefix(self, metrics, prefix="train"):
        tracked_keys = [
            "loss", "reconstruction_loss", "kl_divergence",
            "latent_mean", "latent_std", "logdet_K",
        ]
        return {
            f"{prefix}/{k}": v for k, v in metrics.items() if k in tracked_keys
        }

class GPVAETrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add this line to fix the error
        self.do_grad_scaling = self.use_apex  # True if using mixed precision (FP16/BF16), else False

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        # elif self.use_apex:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()  # module apex and amp not found
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        return model.compute_loss(model, inputs, return_outputs)