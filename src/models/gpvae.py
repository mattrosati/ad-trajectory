import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_constants import *

class GPVAEConfig(PretrainedConfig):
    r"""
    Configuration class to store settings for a gpvae-style model.

    This includes encoder-decoder architecture components, training-related hyperparameters,
    and data handling arguments typically passed via ModelArguments, DataTrainingArguments,
    and CustomTrainingArguments dataclasses.
    """
    model_type = "gpvae"

    def __init__(
        self,
        latent_size=128,
        num_hidden_layers=2,
        intermediate_size=512,
        decoder_hidden_size=128,
        decoder_num_hidden_layers=10,
        decoder_intermediate_size=512,
        hidden_dropout_prob=0.1,
        use_tanh_decoder=False,
        num_labels=6,
        input_dim = 20,
        sigma = 1.005,
        length_scale = 7.0,
        beta = 0.2,

        # Optional additional decoder settings
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.latent_size = latent_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_tanh_decoder = use_tanh_decoder
        self.num_labels = num_labels
        self.input_dim = input_dim
        self.sigma = sigma
        self.length_scale = length_scale
        self.beta = beta


class GPVAEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = [nn.Linear(config.input_dim, config.intermediate_size), nn.ReLU()]
        for _ in range(config.num_hidden_layers - 1):
            layers.extend([nn.Linear(config.intermediate_size, config.intermediate_size), nn.ReLU()])
        self.backbone = nn.Sequential(*layers)
        self.to_mean = nn.Linear(config.intermediate_size, config.latent_size)
        self.to_logvar = nn.Linear(config.intermediate_size, config.latent_size)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.to_mean(h)
        logvar = self.to_logvar(h)
        return mu, logvar


class GPVAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = [nn.Linear(config.latent_size, config.decoder_hidden_size), nn.ReLU()]
        for _ in range(config.num_hidden_layers - 1):
            layers.extend([nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size), nn.ReLU()])
        layers.append(nn.Linear(config.decoder_hidden_size, config.num_labels))
        self.backbone = nn.Sequential(*layers)

    def forward(self, z):
        return self.backbone(z)


class GPVAEPrior(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lengthscale = config.length_scale
        self.variance = config.sigma

    def cauchy_kernel(self, t1, t2):
        dist_sq = (t1[:, None] - t2[None, :]) ** 2
        return self.variance / (1 + dist_sq / self.lengthscale ** 2)

    def forward(self, times, mu, logvar, alpha = 1e-6):
        # Assume times shape [N], mu/logvar shape [N, D]
        K = self.cauchy_kernel(times, times) + alpha * torch.eye(len(times)).to(mu.device)
        L = torch.linalg.cholesky(K)
        z_gp = L @ torch.randn_like(mu)  # [N, D]
        return z_gp, K


class GPVAEModel(PreTrainedModel):
    config_class = GPVAEConfig

    def __init__(self, config):
        super().__init__(config)
        self.encoder = GPVAEEncoder(config)
        self.decoder = GPVAEDecoder(config)
        self.use_gp_prior = True
        self.gp_prior = GPVAEPrior(config) if self.use_gp_prior else None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, inputs, labels=None, times=None, return_dict=True):
        # inputs: [B, D]
        B, D = inputs.shape
        mu, logvar = self.encoder(inputs)             # [B, latent_size]
        z = self.reparameterize(mu, logvar)           # [B, latent_size]
        recon = self.decoder(z)                       # [B, output_dim]

        K = None
        if self.use_gp_prior and times is not None:
            _, K = self.gp_prior(times.view(-1), mu, logvar)  # full GP prior over N timepoints

        if return_dict:
            return {
                "preds": recon,
                "mu": mu,
                "logvar": logvar,
                "z": z,
                "K": K
            }
        else:
            return recon
    
    
    def gpvae_loss(self, targets, out, mu, sigma, K):
        """
        GP-VAE loss function.

        Args:
            targets:         Original input (N x targets)
            out:   Reconstructed input (N x targets)
            mu:        Latent mean (N x latent_dim)
            sigma:     Latent covariance (N x latent_dim x latent_dim), per timestep
            K:         GP kernel matrix (N x N)
            beta:      KL weight (default 0.2)

        Returns:
            Total loss (scalar), recon loss, KL loss
        """

        beta = self.config.beta
        N, latent_dim = mu.shape

        # Select categorical vars and continuous vars
        # TODO: will likely require fixing, currently it is a hardcoded selection
        category_labels = [i for i in targets if i in categorical_features]
        weight = float(len(category_labels) / self.config.num_labels)

        regression_out = out[:, :-1]
        regression_targets = targets[:, :-1]

        classification_logits = out[:, :-1]
        classification_targets = targets[:, :-1]

        loss_reg = torch.nn.functional.mse_loss(regression_out, regression_targets)
        loss_cls = torch.nn.functional.cross_entropy(classification_logits, classification_targets)
        recon_loss = loss_reg + (loss_cls * weight)

        # GP prior covariance for all dimensions: kron(I_d, K)
        K_inv = torch.linalg.inv(K)
        log_det_K = torch.logdet(K)

        # KL divergence
        kl = 0.0
        for d in range(latent_dim):
            mu_d = mu[:, d]             # shape: (N,)
            sigma_d = sigma[:, d, d]    # assuming diagonal covariance

            trace_term = torch.sum(K_inv * torch.diag(sigma_d))
            mahalanobis = mu_d @ K_inv @ mu_d
            log_det_sigma = torch.sum(torch.log(sigma_d))

            kl_d = 0.5 * (trace_term + mahalanobis - N + log_det_K - log_det_sigma)
            kl += kl_d

        total_loss = recon_loss + beta * kl

        return total_loss, recon_loss, kl

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss function for Trainer using [B, D] inputs.
        """
        x = inputs["inputs"]         # [B, D]
        times = inputs["times"]      # [B]
        targets = inputs.get("labels", x)

        outputs = self.forward(inputs=x, labels=targets, times=times)

        x_recon = outputs["preds"]   # [B, D]
        mu = outputs["mu"]           # [B, latent_dim]
        logvar = outputs["logvar"]   # [B, latent_dim]
        sigma = torch.diag_embed(torch.exp(logvar))  # [B, latent_dim, latent_dim]
        K = outputs["K"]             # [B, B]

        loss, recon_loss, kl = self.gpvae_loss(
            targets, x_recon, mu, sigma, K
        )

        outputs_dict = {
            "loss": loss,
            "reconstruction_loss": recon_loss,
            "kl_divergence": kl,
            "latent_mean": mu.mean().item(),
            "latent_std": mu.std().item(),
            "logdet_K": torch.logdet(K).item() if K is not None else 0.0,
        }

        if return_outputs:
            return loss, outputs_dict
        else:
            return loss

    def conditional_gp_posterior(self, z_cond, t_cond, t_query):
        """
        Sample z(t_query) from a conditional GP given z(t_cond).

        Args:
            z_cond (Tensor): known latent vector at t_cond, shape [latent_dim]
            t_cond (float or Tensor): conditioning time (scalar)
            t_query (Tensor): timepoints to predict at, shape [T]

        Returns:
            mu: [T, latent_dim] posterior means
            std: [T, latent_dim] posterior std deviations (sqrt of diag of cov)
        """
        device = z_cond.device
        T = t_query.shape[0]
        D = z_cond.shape[0]

        t_cond = torch.tensor([t_cond], dtype=torch.float32, device=device) if not torch.is_tensor(t_cond) else t_cond.view(1)
        K_cc = self.gp_prior.cauchy_kernel(t_cond, t_cond) + 1e-6 * torch.eye(1, device=device)  # [1,1]
        K_qc = self.gp_prior.cauchy_kernel(t_query, t_cond)                                     # [T,1]
        K_qq = self.gp_prior.cauchy_kernel(t_query, t_query) + 1e-6 * torch.eye(T, device=device)  # [T,T]

        K_cc_inv = torch.inverse(K_cc)
        mu_cond = K_qc @ K_cc_inv @ z_cond.view(1, -1)            # [T, latent_dim]
        cov_cond = K_qq - K_qc @ K_cc_inv @ K_qc.T                # [T, T]

        L = torch.linalg.cholesky(cov_cond)
        eps = torch.randn(T, D, device=device)
        z_query = mu_cond + L @ eps                               # [T, latent_dim]

        std = torch.sqrt(torch.clamp(torch.diagonal(cov_cond, dim1=0, dim2=1), min=1e-6)).unsqueeze(1)  # [T, 1]
        std = std.expand(T, D)  # [T, latent_dim] — shared std across all dimensions

        return mu_cond, std


    def predict_target_trajectory(self, x_cond, t_query, target_idx, t_cond):
        """
        Predict target variable's trajectory with GP mean ± std.

        Returns:
            y_mean: [T]
            y_std: [T]
        """
        device = x_cond.device
        x_cond = x_cond.to(device)
        t_query = t_query.to(device)

        mu_cond, logvar_cond = self.encoder(x_cond.unsqueeze(0))
        z_cond = self.reparameterize(mu_cond, logvar_cond).squeeze(0)  # [latent_dim]

        z_mu, z_std = self.conditional_gp_posterior(z_cond, t_cond, t_query)  # [T, latent_dim]

        x_mu = self.decoder(z_mu)              # [T, num_labels]
        x_std = self.decoder(z_std)            # [T, num_labels]

        y_mean = x_mu[:, target_idx]           # [T]
        y_std = x_std[:, target_idx]           # [T]

        return y_mean, y_std

    @property
    def main_input_name(self):
        return "inputs"
