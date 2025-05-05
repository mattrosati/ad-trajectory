import torch
import torch.nn as nn
import gpytorch

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

    def forward(self, x):
        h = self.net(x)
        mu, logvar = h.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.net(z)


class GPVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

        self.gp_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x, t):
        z, mu, logvar = self.encoder(x)
        recon = self.decoder(z)

        # GP prior: assumes z is [batch, time, latent_dim]
        t_norm = (t - t.mean()) / t.std()
        cov = self.gp_kernel(t_norm).evaluate()
        prior = torch.distributions.MultivariateNormal(
            torch.zeros_like(z), covariance_matrix=cov
        )
        kl_gp = torch.distributions.kl.kl_divergence(
            torch.distributions.MultivariateNormal(mu, torch.diag_embed(torch.exp(logvar))),
            prior
        ).mean()

        return recon, mu, logvar, kl_gp