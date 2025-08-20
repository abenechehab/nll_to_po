"""Train a single policy with the specified loss function.

This function creates a deep copy of the input policy and trains it using the provided
loss function and data. It supports both training and validation phases, with optional
Weights & Biases logging for metrics tracking.

Args:
    policy (MLPPolicy): The base policy to copy and train.
    train_dataloader (torch.utils.data.DataLoader): DataLoader containing training data
        with (X, y) batches.
    loss_function (L.LossFunction): Loss function implementing compute_loss method.
    n_updates (int, optional): Number of training epochs. Defaults to 1.
    learning_rate (float, optional): Learning rate for Adam optimizer. Defaults to 0.001.
    val_dataloader (Optional[torch.utils.data.DataLoader], optional): DataLoader for
        validation data. If None, validation is skipped. Defaults to None.
    wandb_run (optional): Weights & Biases run object for logging metrics. If None,
        no logging is performed. Defaults to None.

Returns:
    MLPPolicy: A trained copy of the input policy.

Notes:
    - The function applies gradient clipping with max_norm=1e9
    - Training and validation metrics are logged to wandb if wandb_run is provided
    - Only scalar metrics (int/float) are logged to wandb
    - The original policy is not modified (deep copy is used)
"""

import copy
from datetime import datetime
import logging
import os
import random
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

import nll_to_po.training.loss as L
from nll_to_po.models.dn_policy import MLPPolicy

from torch.utils.tensorboard import SummaryWriter


def train_single_policy(
    policy: MLPPolicy,
    loss_function: L.LossFunction,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    n_updates: int = 1,
    learning_rate: float = 0.001,
    wandb_run=None,
    tensorboard_writer: Optional[SummaryWriter] = None,
    logger: Optional[logging.Logger] = None,
    scheduler_patience: int = 10,
    early_stopping_patience: int = 20,
    scheduler_factor: float = 0.5,
    min_lr: float = 1e-6,
    device: torch.device = torch.device("cpu"),
):
    """Train a single policy with the specified loss function"""

    trained_policy = copy.deepcopy(policy).train().to(device)
    optimizer = torch.optim.Adam(trained_policy.parameters(), lr=learning_rate)

    # Add scheduler for learning rate reduction on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=min_lr,
    )

    # Initialize metric tracking dictionaries
    train_metrics_history = {}
    val_metrics_history = {}

    # Early stopping variables
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    early_stopped = False

    # Training loop
    if logger is not None:
        logger.info(f"Starting training for {n_updates} epochs")
    for epoch in tqdm(range(n_updates), desc="Training epochs"):
        # Training phase
        trained_policy.train()
        epoch_loss = 0
        epoch_grad_norm = 0

        batch_count = 0
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            loss, metrics = loss_function.compute_loss(trained_policy, X, y)

            optimizer.zero_grad()
            loss.backward()

            # Compute gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trained_policy.parameters(), max_norm=1e9
            )

            optimizer.step()

            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_grad_norm += grad_norm.item()

            batch_count += 1

        # Store averaged training metrics
        avg_loss = epoch_loss / batch_count
        avg_grad_norm = epoch_grad_norm / batch_count

        # Track training metrics
        scalar_metrics = {
            k: v for k, v in metrics.items() if isinstance(v, (int, float))
        }
        scalar_metrics["loss"] = avg_loss
        scalar_metrics["grad_norm"] = avg_grad_norm

        # Add to training metrics history
        for k, v in scalar_metrics.items():
            if k not in train_metrics_history:
                train_metrics_history[k] = []
            train_metrics_history[k].append(v)

        if wandb_run is not None:
            wandb_metrics = {f"train/{k}": v for k, v in scalar_metrics.items()}
            wandb_metrics["epoch"] = epoch
            wandb_metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
            wandb_run.log(wandb_metrics)

        if tensorboard_writer is not None:
            for k, v in scalar_metrics.items():
                tensorboard_writer.add_scalar(f"train/{k}", v, epoch)
            tensorboard_writer.add_scalar(
                "learning_rate", optimizer.param_groups[0]["lr"], epoch
            )

        scheduler_and_early_stopping_loss = avg_loss
        # Validation phase
        if val_dataloader is not None:
            trained_policy.eval()
            val_epoch_loss = 0
            val_batch_count = 0

            with torch.no_grad():
                for X, y in val_dataloader:
                    X, y = X.to(device), y.to(device)
                    val_loss, metrics = loss_function.compute_loss(trained_policy, X, y)
                    val_epoch_loss += val_loss.item()
                    val_batch_count += 1

            avg_val_loss = val_epoch_loss / val_batch_count

            # Track validation metrics
            val_scalar_metrics = {
                k: v for k, v in metrics.items() if isinstance(v, (int, float))
            }
            val_scalar_metrics["loss"] = avg_val_loss

            # Add to validation metrics history
            for k, v in val_scalar_metrics.items():
                if k not in val_metrics_history:
                    val_metrics_history[k] = []
                val_metrics_history[k].append(v)

            if wandb_run is not None:
                wandb_val_metrics = {
                    f"val/{k}": v for k, v in val_scalar_metrics.items()
                }
                wandb_run.log(wandb_val_metrics)

            if tensorboard_writer is not None:
                for k, v in val_scalar_metrics.items():
                    tensorboard_writer.add_scalar(f"val/{k}", v, epoch)

            scheduler_and_early_stopping_loss = avg_val_loss

        # If no validation data, use training loss for scheduler
        scheduler.step(scheduler_and_early_stopping_loss)

        # Early stopping check
        if scheduler_and_early_stopping_loss < best_val_loss:
            best_val_loss = scheduler_and_early_stopping_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            if logger is not None:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            early_stopped = True
            break

    if logger is not None and not early_stopped:
        logger.info(f"Training completed after {n_updates} epochs")

    return trained_policy, train_metrics_history, val_metrics_history


def setup_logger(
    logger_name,
    exp_name,
    log_dir,
    env_id,
    log_level: str = "INFO",
    create_ts_writer: bool = True,
) -> tuple:
    # Clear existing handlers
    root = logging.getLogger(logger_name)
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, env_id, f"{exp_name}")
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, f"{timestamp}.log")

    # Set format for both handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Set up root logger
    root.setLevel(getattr(logging, log_level))
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Set up TensorBoard logger
    writer = None
    if create_ts_writer:
        try:
            tb_log_dir = os.path.join(run_dir, "tensorboard", timestamp)
            os.makedirs(tb_log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_log_dir)
        except ImportError:
            root.warning(
                "Failed to import TensorBoard. "
                "No TensorBoard logging will be performed."
            )

    return root, run_dir, writer


def set_seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
