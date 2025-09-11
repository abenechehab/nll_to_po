import torch
import torch.nn as nn
import torch.nn.functional as F


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
        is_diagonal: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        if is_diagonal:
            # Only diagonal elements can have gradients
            self.matrix = nn.Parameter(torch.eye(input_dim), requires_grad=True)
            # Register a hook to zero out gradients of off-diagonal elements
            self.matrix.register_hook(
                lambda grad: grad * torch.eye(input_dim, device=grad.device)
            )
        else:
            self.matrix = nn.Parameter(torch.eye(input_dim), requires_grad=True)

    def forward(self, state):
        """Forward pass to compute mean and standard deviation."""
        y, y_hat = state[..., : self.input_dim], state[..., self.input_dim :]
        return -torch.einsum("...i,ij,...j->...", (y - y_hat), self.matrix, (y - y_hat))


class RewardMLPMahalanobisDiag(nn.Module):
    def __init__(
        self,
        input_dim: int,
        init_param: float = 1.0,
        use_softplus: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.param = nn.Parameter(init_param * torch.ones(1))
        self.use_softplus = use_softplus

    def forward(self, state):
        """Forward pass to compute mean and standard deviation."""
        y, y_hat = state[..., : self.input_dim], state[..., self.input_dim :]
        param = (
            torch.nn.functional.softplus(self.param)
            if self.use_softplus
            else self.param
        )
        matrix = torch.diag(
            param * torch.ones(self.input_dim, device=self.param.device)
        )
        return -torch.einsum("...i,ij,...j->...", (y - y_hat), matrix, (y - y_hat))


class EmbeddingReward(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 8,
        hidden_sizes: list = [64],
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            *[
                layer
                for h_idx in range(len(hidden_sizes) - 1)
                for layer in (
                    nn.Linear(hidden_sizes[h_idx], hidden_sizes[h_idx + 1]),
                    nn.ReLU(),
                )
            ],
            nn.Linear(hidden_sizes[-1], embedding_dim),
        )

    def forward(self, y, yhat):
        e_y = self.encoder(y)
        e_yhat = self.encoder(yhat)
        sim = F.cosine_similarity(e_y, e_yhat, dim=-1)
        return sim


class EmbeddingMahalanobisReward(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 8,
        hidden_sizes: list = [64],
        train_encoder: bool = True,
        train_matrix: bool = True,
    ):
        super().__init__()
        assert train_encoder or train_matrix, (
            "At least one of train_encoder or train_matrix must be True"
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            *[
                layer
                for h_idx in range(len(hidden_sizes) - 1)
                for layer in (
                    nn.Linear(hidden_sizes[h_idx], hidden_sizes[h_idx + 1]),
                    nn.ReLU(),
                )
            ],
            nn.Linear(hidden_sizes[-1], embedding_dim),
        )
        self.matrix = nn.Parameter(torch.eye(embedding_dim), requires_grad=train_matrix)

        # Freeze encoder parameters if not training
        if not train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, y, yhat):
        e_y = self.encoder(y)
        e_yhat = self.encoder(yhat)
        diff = e_y - e_yhat
        return -torch.einsum("...i,ij,...j->...", diff, self.matrix, diff)
