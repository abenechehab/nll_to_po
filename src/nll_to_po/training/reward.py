"""Reward functions for training policies with PG."""

from abc import ABC, abstractmethod

import torch


class RewardFunction(ABC):
    """Abstract base class for reward functions"""

    name: str

    @abstractmethod
    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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


###classification losses with onehote encoding (on peut en discuter lundi)


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
