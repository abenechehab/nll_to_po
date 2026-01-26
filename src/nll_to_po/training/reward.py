"""Reward functions for training policies with PG."""

from abc import ABC, abstractmethod
import re
from typing import List, Optional

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


def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r".*?<answer>.*?</answer>(?![\s\S])"
    matches = [
        re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions
    ]
    return [1.0 if match else 0.0 for match in matches]


def equation_reward_func(completions, target, nums, verbose=0, **kwargs):
    """
    Evaluates completions based on Mathematical correctness of the answer
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Available numbers
    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, gt, numbers in zip(completions, target, nums):
        try:
            # Check if the format is correct
            match = re.search(r"<answer>(.*?)<\/answer>", completion)
            if match is None:
                rewards.append(0.0)
                if verbose > 0:
                    print("No match found in completion:", completion)
                continue
            # Extract the "answer" part from the completion
            equation = match.group(1).strip()
            if "=" in equation:
                equation = equation.split("=")[0]
            # Extract all numbers from the equation
            used_numbers = [int(n) for n in re.findall(r"\d+", equation)]
            # Check if all numbers are used exactly once
            if sorted(used_numbers) != sorted(numbers):
                rewards.append(0.0)
                if verbose > 0:
                    print(
                        f"Used numbers {used_numbers} do not match available numbers {numbers}."
                    )
                continue
            # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation):
                rewards.append(0.0)
                if verbose > 0:
                    print("Equation contains invalid characters:", equation)
                continue
            # Evaluate the equation with restricted globals and locals
            result = eval(equation, {"__builti'ns__": None}, {})
            # Check if the equation is correct and matches the ground truth
            if abs(float(result) - float(gt)) < 1e-5:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
                if verbose > 0:
                    print(
                        f"Evaluated result {result} does not match ground truth {gt}."
                    )
        except Exception as e:
            if verbose > 0:
                print(f"Exception during evaluation: {e}")
            # If evaluation fails, reward is 0
            rewards.append(0.0)
    return rewards


# def bert_embedding_reward_func(completions, answer, **kwargs):
#     bert_embedder = RM.BertEmbeddingMahalanobisReward(
#         train_encoder=False,
#         train_matrix=False,
#         max_length=2048,
#     )
#     rewards = []
#     for completion, a in zip(completions, answer):
#         try:
#             predicted_answer = re.search(r"<answer>\s*(.*?)\s*</answer>", completion)
#             if predicted_answer is None:
#                 rewards.append(-100.0)
#                 continue
#             reward = bert_embedder(predicted_answer.group(1).strip(), a)
#             rewards.append(reward.item())
#         except Exception:
#             rewards.append(-100.0)
#     return rewards


def embedding_reward_func_constructor(
    model: str,
    U_star=None,
    pooling: str = "mean",
    verbose: int = 0,
    dataset: str = "",
    max_length: int = 2048,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model == "bert":
        bert_embedder = RM.BertEmbeddingMahalanobisReward(
            train_encoder=False,
            train_matrix=False,
            max_length=max_length,
            pooling=pooling,
        )
        min_reward = -100.0
    else:
        bert_embedder = RM.AutoModelEmbeddingMahalanobisReward(
            model_name=model,
            train_encoder=False,
            train_matrix=False,
            max_length=max_length,
            pooling=pooling,
        )
        min_reward = -300.0
    if U_star is not None:
        bert_embedder.set_matrix(matrix=torch.nn.Parameter(U_star))
    bert_embedder.to(device)

    if "Countdown" in dataset:

        def reward_func(completions, answer: List[str], nums, **kwargs):
            rewards = []
            for completion, a, numbers in zip(completions, answer, nums):
                try:
                    # Check if the format is correct
                    predicted_answer = re.search(r"<answer>(.*?)<\/answer>", completion)
                    if predicted_answer is None:
                        rewards.append(min_reward)
                        continue
                    # Extract the "answer" part from the completion
                    equation = predicted_answer.group(1).strip()
                    if "=" in equation:
                        equation = equation.split("=")[0]
                    # Extract all numbers from the equation
                    used_numbers = [int(n) for n in re.findall(r"\d+", equation)]
                    # Check if all numbers are used exactly once
                    if sorted(used_numbers) != sorted(numbers):
                        rewards.append(min_reward)
                        continue
                    # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                    allowed_pattern = r"^[\d+\-*/().\s]+$"
                    if not re.match(allowed_pattern, equation):
                        rewards.append(min_reward)
                        continue
                    reward = bert_embedder(
                        y_hat=predicted_answer.group(1).strip(), y=a.strip()
                    )
                    rewards.append(reward.item())
                except Exception as e:
                    # If evaluation fails
                    if verbose > 0:
                        print(f"completion: {completion}, answer: {a}. \nError: {e}")
                    rewards.append(min_reward)
            return rewards
    else:

        def reward_func(completions, answer, **kwargs):
            rewards = []
            for completion, a in zip(completions, answer):
                reward = bert_embedder(y_hat=completion.strip(), y=a.strip())
                rewards.append(reward.item())
            return rewards

    return reward_func


def bert_embedding_reward_func_gsm8k(completions, answer, rationale, **kwargs):
    min_reward = -100.0
    bert_embedder = RM.BertEmbeddingMahalanobisReward(
        train_encoder=False,
        train_matrix=False,
        max_length=2048,
    )
    rewards = []
    for completion, a, r in zip(completions, answer, rationale):
        try:
            full_answer = f"{r}\nFinal Answer: {a}"
            reward = bert_embedder(completion.strip(), full_answer.strip())
            rewards.append(reward.item())
        except Exception:
            # If evaluation fails
            rewards.append(min_reward)
    return rewards
