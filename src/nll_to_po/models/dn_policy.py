"""Basic density network policy model."""

from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicy(nn.Module):
    """Multi-layer perceptron policy model with configurable architecture."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int],
        fixed_logstd: bool = False,
        use_batchnorm: bool = False,
        use_layernorm: bool = False,
    ):
        """Initialize the MLPPolicy.

        Args:
            input_dim (int): input dimension.
            output_dim (int): output dimension.
            hidden_sizes (list): list of hidden layer sizes.
            fixed_logstd (bool, optional): whether to use a fixed log standard deviation. Defaults to False.
            use_batchnorm (bool, optional): whether to use batch normalization. Defaults to False.
            use_layernorm (bool, optional): whether to use layer normalization. Defaults to False.
        """

        super().__init__()
        assert not (use_batchnorm and use_layernorm), (
            "Cannot use both batchnorm and layernorm."
        )
        layers = []
        dims = [input_dim] + hidden_sizes
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            if use_layernorm:
                layers.append(nn.LayerNorm(dims[i + 1]))
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


class MulticlassLogisticRegression(nn.Module):
    """Multiclass logistic regression policy."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        logits = self.linear(x)  # (B, C)
        probs = F.softmax(logits, dim=-1)  # (B, C)
        return logits, probs


class MulticlassMLP(nn.Module):
    """Multiclass MLP policy."""

    def __init__(
        self, input_dim=785, hidden_sizes: List[int] = [512, 256], output_dim=10
    ):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_sizes
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.logits = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        x = self.net(x)
        logits = self.logits(x)
        return logits, F.softmax(logits, dim=-1)


class MLPPolicyBounded(nn.Module):
    """Multi-layer perceptron policy with Bounded log_std (learned bounds)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: list,
        fixed_logstd: bool = False,
        log_var_max: Union[float, torch.Tensor] = 1.6,
        log_var_min: Union[float, torch.Tensor] = -10.0,
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

        self.fixed_logstd = fixed_logstd
        if fixed_logstd:
            self.log_std = nn.Parameter(torch.zeros(output_dim))
        else:
            self.log_std = nn.Sequential(
                nn.Linear(dims[-1], output_dim),
            )

            # Learnable logsigma bounds:
            self.max_logsigma = nn.Parameter(torch.ones([1, output_dim]) * log_var_max)
            self.min_logsigma = nn.Parameter(-torch.ones([1, output_dim]) * log_var_min)

    def forward(self, state):
        """Forward pass to compute mean and standard deviation."""
        common = self.net(state)
        mean = self.mean(common)
        log_std = self.log_std(common) if not self.fixed_logstd else self.log_std

        # Bound logsigma (following PETS' implementation)
        std = self.max_logsigma - nn.Softplus()(self.max_logsigma - log_std)
        std = self.min_logsigma + nn.Softplus()(std - self.min_logsigma)

        std = torch.exp(log_std)
        return mean, std
