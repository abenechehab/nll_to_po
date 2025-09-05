import torch
import torch.nn as nn


class RewardMLP(nn.Module):
    """Multi-layer perceptron reward network with configurable architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list,
    ):
        super().__init__()
        layers = []
        dims = [2 * input_dim] + hidden_sizes
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

        self.reward = nn.Sequential(
            nn.Linear(dims[-1], 1),
        )

        # sigma(y_hat^T W + b)
        # sigma((y-y_hat)^T W)

    def forward(self, state):
        """Forward pass to compute mean and standard deviation."""
        common = self.net(state)
        reward = self.reward(common)
        return reward.squeeze(-1)


class RewardMLPMahalanobis(nn.Module):
    """Multi-layer perceptron reward network with configurable architecture."""

    def __init__(
        self,
        input_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.matrix = nn.Parameter(torch.eye(input_dim), requires_grad=True)

        # sigma(y_hat^T W + b)
        # sigma((y-y_hat)^T W)

    def forward(self, state):
        """Forward pass to compute mean and standard deviation."""
        y, y_hat = state[..., : self.input_dim], state[..., self.input_dim :]
        return -torch.einsum("...i,ij,...j->...", (y - y_hat), self.matrix, (y - y_hat))
