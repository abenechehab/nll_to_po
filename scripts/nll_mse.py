# import os

import torch

from nll_to_po.models.dn_policy import MLPPolicy
from nll_to_po.training.utils import train_single_policy
import nll_to_po.training.loss as L

import wandb

import logging
import tyro
from dataclasses import dataclass
from typing import List

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

# os.environ["WANDB_SILENT"] = "true"  # Suppress WandB output


# --------------


@dataclass
class ExperimentConfig:
    wandb_project: str = "new_nllpo"
    n_experiments: int = 100
    input_dim: int = 4
    output_dim: int = 2
    hidden_sizes: List[int] = [64, 64]
    learning_rate: float = 0.001
    fixed_logstd: bool = False
    init_dist_scale: float = 0.5
    init_dist_n_samples_list: List[int] = [1, 25]
    n_updates_list: List[int] = [100, 500]
    init_dist_loc_list: List[float] = [1.0, 7.0]

    def __post_init__(self):
        assert len(self.n_updates_list) == len(
            self.init_dist_loc_list
        ), "n updates must match init dist loc"


def main(args: ExperimentConfig):
    for init_dist_n_samples in args.init_dist_n_samples_list:
        for n_updates, init_dist_loc in zip(
            args.n_updates_list, args.init_dist_loc_list
        ):
            for loss_function in [L.MSE(), L.NLL()]:
                for _ in range(args.n_experiments):
                    policy = MLPPolicy(
                        args.input_dim,
                        args.output_dim,
                        args.hidden_sizes,
                        args.fixed_logstd,
                    )

                    # Generate new random data for each experiment
                    X = torch.randn(1, args.input_dim)
                    mean_y = torch.ones((1, args.output_dim)) * init_dist_loc
                    y = (
                        mean_y
                        + torch.randn(init_dist_n_samples, args.output_dim)
                        * args.init_dist_scale
                    )
                    X = X.repeat(init_dist_n_samples, 1)  # Repeat X for each sample

                    batch_size = X.shape[0]

                    # Create a DataLoader
                    train_dataset = torch.utils.data.TensorDataset(X, y)
                    train_dataloader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=batch_size, shuffle=True
                    )

                    wandb_run = wandb.init(
                        project=args.wandb_project,
                        name=loss_function.name,
                        config={
                            "batch_size": batch_size,
                            "fixed_logstd": args.fixed_logstd,
                            "init_dist_loc": init_dist_loc,
                            "init_dist_scale": args.init_dist_scale,
                            "init_dist_n_samples": init_dist_n_samples,
                            "learning_rate": args.learning_rate,
                        },
                    )

                    # Run comparison
                    train_single_policy(
                        policy=policy,
                        train_dataloader=train_dataloader,
                        loss_function=loss_function,
                        n_updates=n_updates,
                        learning_rate=args.learning_rate,
                        wandb_run=wandb_run,
                    )

                    wandb_run.finish()


if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args=args)
