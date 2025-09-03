"""Loss functions for training policies in NLL to PO framework."""

from abc import ABC, abstractmethod

from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

import nll_to_po.training.reward as R

if TYPE_CHECKING:
    from nll_to_po.models.dn_policy import MLPPolicy


class LossFunction(ABC):
    """Abstract base class for loss functions"""

    name: str

    @abstractmethod
    def compute_loss(
        self, policy: "MLPPolicy", X: torch.Tensor, y: torch.Tensor, mu: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Compute the loss given policy, inputs X, and targets y"""
        pass


class MSE(LossFunction):
    """MSE loss using only the mean prediction"""

    name = "MSE"

    def compute_loss(self, policy, X, y, mu, std):
        mean, _ = policy(X)
        loss = nn.MSELoss()(mean, y)
        return loss, {
            "mean_error": torch.sqrt(nn.MSELoss()(mean, mu)).item(),
            "loss": loss.item(),
        }


class NLL(LossFunction):
    """Negative log-likelihood loss"""

    name = "NLL"

    def compute_loss(self, policy, X, y, mu, std):
        mean, sigma = policy(X)
        dist = D.Normal(mean, sigma)
        nll = -dist.log_prob(y).mean()
        metrics = {
            "mean_error": torch.sqrt(nn.MSELoss()(mean, mu)).item(),
            "NLL": nll.item(),
            "dist": D.Normal(mean[0].clone(), sigma[0].clone()),
            "loss": nll.item(),
            "sigma_error": torch.norm(sigma - std, p="fro", dim=-1).mean().item(),
            "entropy": dist.entropy().mean().item(),
        }
        if sigma.shape[-1] == 2:
            for idx in range(sigma.shape[-1]):
                if policy.fixed_logstd:
                    metrics[f"std_{idx}"] = sigma[idx].mean().item()
                else:
                    metrics[f"std_{idx}"] = sigma[:, idx].mean().item()
        return nll, metrics


class PG(LossFunction):
    """Policy gradient loss with configurable reward and entropy regularization"""

    name = "PG"

    def __init__(
        self,
        reward_fn: R.RewardFunction,
        n_generations: int = 5,
        use_rsample: bool = False,
        reward_transform: str = "none",  # "normalize", "rbf", "none"
        rbf_gamma: float = 1.0,
        entropy_weight: float = 0.01,
        clip_coef: Optional[float] = None,
    ):
        self.n_generations = n_generations
        self.use_rsample = use_rsample
        self.reward_transform = reward_transform
        self.rbf_gamma = rbf_gamma
        self.entropy_weight = entropy_weight
        self.reward_fn = reward_fn
        self.name = f"{self.name}(lam={self.entropy_weight})_{self.reward_fn.name}"
        self.clip_coef = clip_coef

    def _transform_rewards(self, rewards):
        """Apply reward transformation"""
        if self.reward_transform == "rbf":
            return torch.exp(self.rbf_gamma * rewards)
        elif self.reward_transform == "normalize":
            rewards_min, _ = rewards.clone().detach().aminmax(dim=0, keepdim=True)
            return rewards - rewards_min
        else:  # "none"
            return rewards

    def compute_loss(self, policy, X, y, mu, std):
        mean, sigma = policy(X)
        dist = D.Normal(mean, sigma)

        if self.use_rsample:
            samples = dist.rsample((self.n_generations,))
            rewards = self.reward_fn(y_hat=samples, y=y)
            rewards = self._transform_rewards(rewards)
            loss = -rewards.mean()
        else:
            samples = dist.sample((self.n_generations,))
            neg_log_prob = -dist.log_prob(samples).mean(dim=-1)
            rewards = self.reward_fn(y_hat=samples, y=y)
            rewards = self._transform_rewards(rewards)
            if self.clip_coef is not None:
                neg_log_prob = torch.clamp(
                    neg_log_prob, -self.clip_coef, self.clip_coef
                )
            loss = (neg_log_prob * rewards).mean()

        loss -= self.entropy_weight * dist.entropy().mean()

        metrics = {
            "mean_error": torch.sqrt(nn.MSELoss()(mean, mu)).item(),
            "NLL": -dist.log_prob(y).mean().item(),
            "dist": D.Normal(mean[0].clone(), sigma[0].clone()),
            "entropy": dist.entropy().mean().item(),
            "reward_mean": rewards.mean().item(),
            "loss": loss.item(),
            "sigma_error": torch.norm(sigma - std, p="fro", dim=-1).mean().item(),
        }
        if sigma.shape[-1] == 2:
            for idx in range(sigma.shape[-1]):
                if policy.fixed_logstd:
                    metrics[f"std_{idx}"] = sigma[idx].mean().item()
                else:
                    metrics[f"std_{idx}"] = sigma[:, idx].mean().item()
        return loss, metrics


###losses pour le cas multivarie elles sont toutes notee {nom original}_Full_Cov


class NLL_Full_Cov(LossFunction):
    """Negative log-likelihood loss"""

    name = "NLL_Full_Cov"

    def __init__(
        self, target_mu: torch.Tensor = None, target_sigma: torch.Tensor = None
    ):
        super().__init__()
        self.target_mu = target_mu
        self.target_sigma = target_sigma

    def compute_loss(self, policy, X, y):
        # std ici c est scale_tril la triangulaire inf c
        mean, std = policy(X)
        # dist = torch.distributions.Normal(mean, std)
        dist = D.MultivariateNormal(
            mean, scale_tril=std
        )  # car je considere des matrices pleines
        cholesky_part = torch.linalg.cholesky(self.target_sigma)
        metrics = {
            "mean_error": nn.MSELoss()(mean.mean(dim=0), y.mean(dim=0)).item(),
            "L2_error": torch.norm(mean.mean(dim=0) - self.target_mu, p=2).item(),
            "std_error": torch.norm(std[0] - cholesky_part, p="fro").item(),
            "NLL": -dist.log_prob(y).mean().item(),
            "dist": D.MultivariateNormal(
                loc=mean[0].clone(), scale_tril=std[0].clone()
            ),
            "sigma_norm": torch.norm(std, p="fro", dim=-1).mean().item(),
        }
        return -dist.log_prob(y).mean(), metrics


class PG_Full_Cov(LossFunction):
    """Policy optimization loss with configurable reward and entropy regularization"""

    name = "PG_Full_Cov"

    def __init__(
        self,
        reward_fn: R.RewardFunction,
        n_generations: int = 5,
        use_rsample: bool = False,
        reward_transform: str = "normalize",  # "normalize", "rbf", "none"
        rbf_gamma: Optional[float] = None,
        entropy_weight: float = 0.01,
        target_mu: torch.Tensor = None,
        target_sigma: torch.Tensor = None,
    ):
        self.n_generations = n_generations
        self.use_rsample = use_rsample
        self.reward_transform = reward_transform
        self.rbf_gamma = rbf_gamma
        self.entropy_weight = entropy_weight
        self.reward_fn = reward_fn
        self.target_mu = target_mu
        self.target_sigma = target_sigma

    def _transform_rewards(self, rewards):
        """Apply reward transformation"""
        if self.reward_transform == "rbf" and self.rbf_gamma is not None:
            return torch.exp(self.rbf_gamma * rewards)
        elif self.reward_transform == "normalize":
            rewards_min, _ = rewards.aminmax(dim=0, keepdim=True)
            return rewards - rewards_min
        else:  # "none"
            return rewards

    def compute_loss(self, policy, X, y):
        mean, std = policy(X)
        # dist = torch.distributions.Normal(mean, std)
        dist = D.MultivariateNormal(mean, scale_tril=std)
        if self.use_rsample:
            samples = dist.rsample((self.n_generations,))
            rewards = self.reward_fn(y_hat=samples, y=y)
            rewards = self._transform_rewards(rewards)
            loss = -rewards.mean()
        else:
            samples = dist.sample((self.n_generations,))
            neg_log_prob = -dist.log_prob(samples)  # .mean() #dim=-1
            rewards = self.reward_fn(y_hat=samples, y=y)
            rewards = self._transform_rewards(rewards)
            loss = (neg_log_prob * rewards).mean()

        loss -= self.entropy_weight * dist.entropy().mean()  # signe -
        cholesky_lower = torch.linalg.cholesky(self.target_sigma)
        metrics = {
            "mean_error": nn.MSELoss()(mean.mean(dim=0), y.mean(dim=0)).item(),
            "L2_error": torch.norm(mean.mean(dim=0) - self.target_mu, p=2).item(),
            "std_error": torch.norm(std[0] - cholesky_lower, p="fro").item(),
            "NLL": -dist.log_prob(y).mean().item(),
            "dist": D.MultivariateNormal(
                loc=mean[0].clone(), scale_tril=std[0].clone()
            ),
            "entropy": dist.entropy().mean().item(),
            "sigma_norm": torch.norm(std, p="fro", dim=-1).mean().item(),
            "loss": loss.item(),
        }
        return loss, metrics


###Classifications loss###


class NLL_Classification(LossFunction):
    name = "NLL_ClS"

    def compute_loss(self, policy, X, y, mu, std):
        logits, probs = policy(X)
        loss = F.cross_entropy(logits, y)  # (B, C), y: (B,) long

        with torch.no_grad():
            # probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)
            acc = (pred == y).float().mean().item()
            ent = (-(probs * probs.clamp_min(1e-12).log()).sum(-1)).mean().item()

        metrics = {"NLL": loss.item(), "accuracy": acc, "entropy": ent}
        return loss, metrics


class PO_Entropy_Classification(LossFunction):
    """Policy gradient with entropy regularization for classification.
    Policy outputs (logits, probs); distribution is Categorical(logits=logits).
    """

    name = "PO_ENT_CL"

    def __init__(
        self,
        reward_fn: R.RewardFunction,
        n_generations: int = 5,
        use_rsample: bool = False,
        reward_transform: str = "normalize",  # "normalize", "rbf", "none"
        rbf_gamma: Optional[float] = None,
        entropy_weight: float = 0.01,
        temperature: float = 1.0,
    ):
        self.n_generations = n_generations
        self.use_rsample = use_rsample
        self.reward_transform = reward_transform
        self.rbf_gamma = rbf_gamma
        self.entropy_weight = entropy_weight
        self.reward_fn = reward_fn
        self.temperature = temperature

    def _transform_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.reward_transform == "rbf" and self.rbf_gamma is not None:
            return torch.exp(self.rbf_gamma * rewards)
        elif self.reward_transform == "normalize":
            rewards_min, _ = rewards.aminmax(dim=0, keepdim=True)
            return rewards - rewards_min
        else:
            return rewards

    def compute_loss(self, policy, X, y, mu, std):
        """
        policy(X): returns logits and probs of shape (B, C)
        y: LongTensor of shape (B,) with class indices
        """
        logits, probs = policy(X)
        dist = D.Categorical(logits=logits)

        samples = dist.sample((self.n_generations,))
        logp = dist.log_prob(samples)

        y_b = y.unsqueeze(0).expand_as(samples)

        rewards = self.reward_fn(y_hat=samples, y=y_b)
        rewards = self._transform_rewards(rewards)  # (G,B)

        pg_loss = -(logp * rewards).mean()

        ent = dist.entropy().mean()
        loss = pg_loss - self.entropy_weight * ent

        with torch.no_grad():
            nll = F.cross_entropy(logits, y).item()
            acc = (probs.argmax(dim=-1) == y).float().mean().item()

        metrics = {
            "NLL": nll,
            "accuracy": acc,
            "pg_loss": pg_loss.item(),
            "total_loss": loss.item(),
        }
        return loss, metrics
