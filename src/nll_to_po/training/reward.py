"""Reward functions for training policies with PG."""

from abc import ABC, abstractmethod
import re
from typing import Optional

import torch
from torch.func import functional_call

import nll_to_po.models.reward_network as RM


class RewardFunction(ABC):
    """Abstract base class for reward functions"""

    name: str

    @abstractmethod
    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the reward given generation y_hat, and groundtruth y"""
        pass


class Mahalanobis(RewardFunction):
    """Mahalanobis reward: - (y-y_star)^T M (y-y_star)"""

    name = "Mahalanobis"

    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix
        first_diag_element = self.matrix[0, 0]
        desc = (
            r"$\mathrm{I}$"
            if first_diag_element == 1.0
            else r"$\frac{\lambda n}{2 Tr(\Sigma)}\mathrm{I}$"
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


class OneHotMahalanobis(RewardFunction):
    """Mahalanobis reward on one-hot vectors: - (y-y_star)^T M (y-y_star)"""

    name = "OneHotMahalanobis"

    def __init__(self, U: torch.Tensor, num_classes: int):
        self.U = U  # (C, C), SPD
        self.C = num_classes

    def __call__(self, y_hat, y):
        # y_hat, y: (G,B) class ids
        yh = torch.nn.functional.one_hot(y_hat, num_classes=self.C).float()  # (G,B,C)
        # yh=F.softmax(y_hat, dim=-1)
        yt = torch.nn.functional.one_hot(y, num_classes=self.C).float()  # (G,B,C)
        diff = yh - yt  # (G,B,C)
        # - (diff^T U diff) per (g,b)
        return -torch.einsum("gbc,cd,gbd->gb", diff, self.U, diff)


class RewardNetwork(RewardFunction):
    """MLP (or any other) reward function"""

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


class OneHotRewardNetwork(RewardFunction):
    """Applies a given reward network to one hot encodings of y_hat and y"""

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


# #################################
# ******* Reward functions ********
# #################################


def format_reward_func(completions, target, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers

      Returns:
          list[float]: Reward scores
    """
    rewards = []

    for completion, gt in zip(completions, target):
        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion
            # Check if the format is correct
            # regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            regex = r"^<think>[\s\S]*?<\/think>\s*<answer>[\s\S]*?<\/answer>$"
            match = re.search(regex, completion, re.DOTALL)
            # if the format is not correct, reward is 0
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def equation_reward_func(completions, target, nums, **kwargs):
    """
    Evaluates completions based on:
    1. Proper <answer> formatting
    2. Mathematical correctness
    3. Correct usage of provided numbers

    Supports:
    - Parentheses
    - Full formulas with '=' inside <answer>
      e.g. <answer> 72 / (30 - 29) = 72 </answer>
    """
    rewards = []

    for completion, gt, numbers in zip(completions, target, nums):
        try:
            completion = "<think>" + completion

            # Extract answer content
            match = re.search(r"<answer>(.*?)<\/answer>", completion, re.DOTALL)
            if match is None:
                rewards.append(0.0)
                continue

            answer_text = match.group(1).strip()

            # Split on '=' if present
            if "=" in answer_text:
                lhs, rhs = map(str.strip, answer_text.split("=", 1))
            else:
                lhs, rhs = answer_text, None

            # Allowed characters: digits, operators, parentheses, decimal points, whitespace
            allowed_pattern = r"^[\d+\-*/().\s]+$"

            if not re.match(allowed_pattern, lhs):
                rewards.append(0.0)
                continue
            if rhs is not None and not re.match(allowed_pattern, rhs):
                rewards.append(0.0)
                continue

            # Extract numbers ONLY from the left-hand side expression
            used_numbers = [int(n) for n in re.findall(r"\d+", lhs)]

            if sorted(used_numbers) != sorted(numbers):
                rewards.append(0.0)
                continue

            # Safe evaluation
            lhs_value = eval(lhs, {"__builtins__": None}, {})

            # Check RHS consistency if present
            if rhs is not None:
                rhs_value = eval(rhs, {"__builtins__": None}, {})
                if abs(float(lhs_value) - float(rhs_value)) > 1e-5:
                    rewards.append(0.0)
                    continue

            # Final check against ground truth
            if abs(float(lhs_value) - float(gt)) < 1e-5:
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        except Exception:
            rewards.append(0.0)

    return rewards


def bert_embedding_reward_func(completions, target, **kwargs):
    bert_embedder = RM.BertEmbeddingMahalanobisReward(
        train_encoder=False,
        train_matrix=False,
        max_length=2048,
    )
    rewards = []
    for completion, gt in zip(completions, target):
        try:
            reward = bert_embedder(completion, gt)
            rewards.append(reward.item())
        except Exception:
            rewards.append(0.0)
    return rewards
