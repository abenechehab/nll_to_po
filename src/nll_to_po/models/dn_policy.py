"""Basic density network policy model."""

import torch
import torch.nn as nn


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
