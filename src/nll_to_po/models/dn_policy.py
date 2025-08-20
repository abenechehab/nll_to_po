"""Basic density network policy model."""

# test
import torch
import torch.nn as nn
import torch.functional as F


class MLPPolicy(nn.Module):
    """Multi-layer perceptron policy model with configurable architecture."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: list,
        fixed_logstd: bool = False,
    ):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_sizes
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

        self.mean = nn.Sequential(
            nn.Linear(dims[-1], output_dim),
        )
        if fixed_logstd:
            self.log_std = nn.Parameter(torch.zeros(output_dim))
        else:
            self.log_std = nn.Sequential(
                nn.Linear(dims[-1], output_dim),
            )
        self.fixed_logstd = fixed_logstd

    def forward(self, state):
        """Forward pass to compute mean and standard deviation."""
        common = self.net(state)
        mean = self.mean(common)
        log_std = self.log_std(common) if not self.fixed_logstd else self.log_std
        std = torch.exp(log_std)
        return mean, std


class LinearGaussian(nn.Module):
    """Linear Gaussian policy."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        fixed_logstd: bool = False,
    ):
        super().__init__()

        self.mean = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )
        if fixed_logstd:
            self.log_std = nn.Parameter(torch.zeros(output_dim))
        else:
            self.log_std = nn.Sequential(
                nn.Linear(input_dim, output_dim),
            )
        self.fixed_logstd = fixed_logstd

    def forward(self, state):
        """Forward pass to compute mean and standard deviation."""
        mean = self.mean(state)
        log_std = self.log_std(state) if not self.fixed_logstd else self.log_std
        std = torch.exp(log_std)
        return mean, std


### MLP policy for the non iid multivariate case :)


class MLPPolicy_Full_Cov(nn.Module):
    """Sigma = L L^T avec L triangulaire inférieure prédite par le réseau."""

    def __init__(self, input_dim, output_dim, hidden_sizes):
        super().__init__()
        self.output_dim = output_dim
        # tips pour eviter de ralculer les indices a chaque forward.
        self.register_buffer("tril_idx", torch.tril_indices(output_dim, output_dim, 0))
        self.register_buffer("diag_index", torch.arange(output_dim))

        layers, dims = [], [input_dim] + hidden_sizes
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.Tanh()]
        self.net = nn.Sequential(*layers)

        # têtes
        self.mean_layer = nn.Linear(dims[-1], output_dim)
        n_tril = output_dim * (output_dim + 1) // 2
        self.tril_layer = nn.Linear(dims[-1], n_tril)

        # hyperparams de stabilité
        self.min_std = 1e-3
        self.max_std = 1e3
        self.jitter = 1e-6

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_layer(h)

        B, D = x.shape[0], self.output_dim
        scale_tril = h.new_zeros(B, D, D)

        tril_params = self.tril_layer(h)
        scale_tril[:, self.tril_idx[0], self.tril_idx[1]] = tril_params

        d = scale_tril[:, self.diag_index, self.diag_index]  # (B, D)
        d = F.softplus(
            d
        )  # softplus au lieu de exp pour eviter que les valeurs explosent
        d = d.clamp(max=self.max_std)  # je clamp sinon la nll explose
        scale_tril[:, self.diag_index, self.diag_index] = d

        eye = torch.eye(D, device=x.device, dtype=scale_tril.dtype).unsqueeze(0)
        scale_tril = scale_tril + self.jitter * eye

        return mean, scale_tril
