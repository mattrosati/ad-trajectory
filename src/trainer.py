from transformers import Trainer
import torch

class GPVAETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        x, t = inputs["x"], inputs["t"]
        out = model.compute_loss(x, t)
        loss = out["loss"]
        if return_outputs:
            return loss, out
        return loss
