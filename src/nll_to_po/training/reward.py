"""Reward functions for training policies with PG."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch.func import functional_call

import nll_to_po.models.reward_network as RM


class RewardFunction(ABC):
    """Abstract base class for reward functions"""

    name: str

    @abstractmethod
    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the reward given generation y, and groundtruth y_star"""
        pass


class Mahalanobis(RewardFunction):
    """Mahalanobis reward: - (y-y_star)^T M (y-y_star)"""

    name = "Mahalanobis"

    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix
        first_diag_element = self.matrix[0, 0]
        desc = (
            r"$I$"
            if first_diag_element == 1.0
            else r"$\frac{\lambda n}{2 Tr(\Sigma)}I$"
        )
        self.name = f"{self.name}({desc})"

    def __call__(self, y_hat, y):
        y_hat = torch.squeeze(y_hat)
        y = torch.squeeze(y)
        diff = y_hat - y

        if diff.dim() == 3:
            # Handle 3D case: (batch, group, features)
            return -torch.einsum("gbi,ij,gbj->gb", diff, self.matrix, diff)
        elif diff.dim() == 2:
            # Handle 2D case: (batch, features)
            return -torch.einsum("bi,ij,bj->b", diff, self.matrix, diff)
        else:
            raise ValueError(
                f"Expected diff to have 2 or 3 dimensions, got {diff.dim()}"
            )


class OneHotMahalanobis:
    def __init__(self, U: torch.Tensor, num_classes: int):
        self.U = U  # (C, C), SPD
        self.C = num_classes

    def __call__(self, y_hat, y):
        # print(y_hat)
        # y_hat, y: (G,B) class ids
        yh = torch.nn.functional.one_hot(y_hat, num_classes=self.C).float()  # (G,B,C)
        # yh=F.softmax(y_hat, dim=-1)
        yt = torch.nn.functional.one_hot(y, num_classes=self.C).float()  # (G,B,C)
        diff = yh - yt  # (G,B,C)
        # - (diff^T U diff) per (g,b)
        return -torch.einsum("gbc,cd,gbd->gb", diff, self.U, diff)


class OneHotRewardNetwork(RewardFunction):
    """One hot reward network"""

    name = "OneHotRewardNetwork"

    def __init__(self, reward_network: RM.RewardMLP, num_classes: int):
        self.reward_network = reward_network
        self.num_classes = num_classes

    def __call__(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        reward_network: Optional[RM.RewardMLP] = None,
    ):
        if reward_network is not None:
            self.reward_network = reward_network

        # y_hat, y: (G,B) class ids
        one_hot_y_hat = torch.nn.functional.one_hot(
            y_hat, num_classes=self.num_classes
        ).float()  # (G,B,C)
        # yh=F.softmax(y_hat, dim=-1)
        one_hot_y = torch.nn.functional.one_hot(
            y, num_classes=self.num_classes
        ).float()  # (G,B,C)

        input_rn = torch.cat([one_hot_y_hat, one_hot_y], dim=-1)
        return self.reward_network(input_rn)


class FuncOneHotRewardNetwork(RewardFunction):
    """Functional One hot reward network"""

    name = "FuncOneHotRewardNetwork"

    def __init__(self, reward_model, reward_params, num_classes: int):
        self.reward_model = reward_model
        self.reward_params = reward_params
        self.num_classes = num_classes

    def __call__(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        reward_model=None,
        reward_params=None,
    ):
        if reward_model is not None:
            self.reward_model = reward_model
        if reward_params is not None:
            self.reward_params = reward_params

        # y_hat, y: (G,B) class ids
        one_hot_y_hat = torch.nn.functional.one_hot(
            y_hat.long(), num_classes=self.num_classes
        ).float()  # (G,B,C)
        # yh=F.softmax(y_hat, dim=-1)
        one_hot_y = torch.nn.functional.one_hot(
            y.long(), num_classes=self.num_classes
        ).float()  # (G,B,C)

        input_rn = torch.cat([one_hot_y_hat, one_hot_y], dim=-1)
        return functional_call(self.reward_model, self.reward_params, input_rn)


class RewardNetwork(RewardFunction):
    """MLP reward function"""

    name = "RewardNetwork"

    def __init__(self, reward_network: RM.RewardMLP):
        self.reward_network = reward_network

    def __call__(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        reward_network: Optional[RM.RewardMLP] = None,
    ):
        if reward_network is not None:
            self.reward_network = reward_network

        # Ensure both tensors have the same number of dimensions
        if y_hat.dim() == 3 and y.dim() == 2:
            y = y.unsqueeze(0).expand(y_hat.size(0), -1, -1)

        input_rn = torch.cat([y_hat, y], dim=-1)
        return self.reward_network(input_rn)


class FuncRewardNetwork(RewardFunction):
    """Functional MLP reward function"""

    name = "FuncRewardNetwork"

    def __init__(self, reward_model, reward_params):
        self.reward_model = reward_model
        self.reward_params = reward_params

    def __call__(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        reward_model=None,
        reward_params=None,
    ):
        if reward_model is not None:
            self.reward_model = reward_model
        if reward_params is not None:
            self.reward_params = reward_params

        # Ensure both tensors have the same number of dimensions
        if y_hat.dim() == 3 and y.dim() == 2:
            y = y.unsqueeze(0).expand(y_hat.size(0), -1, -1)

        input_rn = torch.cat([y_hat, y], dim=-1)
        return functional_call(self.reward_model, self.reward_params, input_rn)
