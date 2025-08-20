"""
Parallel experiment runner for model-based RL-style supervised training using Ray.

This script is a refactor of the notebook `notebook/mbrl.ipynb` into a CLI.
It:
- Loads Minari datasets and prepares train/val splits
- Builds a policy and trains it under different losses (MSE, NLL, PG variants)
- Runs experiments in parallel with Ray (default 0.1 GPU per task)
- Saves metrics for each run to Parquet for efficient storage

Notes:
- Visualizations are intentionally omitted.
- Requires optional extras: `pip install -e .[mbrl]` (adds minari, ray, pyarrow)
"""

import os
import json
import logging
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

import minari
import ray
import tyro

import nll_to_po.models.dn_policy as Policy
import nll_to_po.training.loss as L
import nll_to_po.training.reward as R
from nll_to_po.training.utils import (
    train_single_policy,
    setup_logger,
    set_seed_everywhere,
)


@dataclass
class ExperimentConfig:
    # Data args
    dataset: str = "mujoco/halfcheetah/medium-v0"
    train_size: float = 0.7
    data_proportion: float = 0.1
    batch_size: int = -1

    # Model/optim args
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    fixed_logstd: bool = False
    n_updates: int = 300
    learning_rate: float = 0.01
    early_stopping_patience: int = 50

    # PG-specific
    n_generations: int = 5
    pg_rsample: bool = False
    reward_transform: str = "normalize"
    entropy_weights: List[float] = field(default_factory=lambda: [0.1, 1.0, 5.0])

    # Infra/logging
    n_experiments: int = 3
    use_wandb: bool = False
    wandb_project: str = "tractable"
    env_id: str = "mbrl"
    log_dir: str = "logs"
    results_dir: str = "logs/mbrl_results"
    ray_address: Optional[str] = None
    merge_results: bool = True


# Reduce noisy logs
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
os.environ["WANDB_SILENT"] = "true"


def generate_data_minari(
    dataset_name: str,
    train_size: float = 0.8,
    data_proportion: float = 1.0,
    batch_size: int = -1,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict[str, int]]:
    """Create train/val DataLoaders from a Minari dataset.

    Returns: (train_loader, val_loader, {input_dim, output_dim})
    """
    seed = np.random.randint(0, 1_000_000)
    set_seed_everywhere(seed=int(seed))

    dataset = minari.load_dataset(dataset_name, download=True)
    dataset.set_seed(seed=int(seed))

    observations = []
    actions = []
    next_observations = []
    for episode in dataset:
        observations.append(episode.observations[:-1])
        actions.append(episode.actions)
        next_observations.append(episode.observations[1:])
    observations = np.concatenate(observations, axis=0)
    actions = np.concatenate(actions, axis=0)
    next_observations = np.concatenate(next_observations, axis=0)

    obs_dim = observations.shape[1]
    action_dim = actions.shape[1]

    X = torch.tensor(
        np.concatenate([observations, actions], axis=1), dtype=torch.float32
    )
    y = torch.tensor(next_observations, dtype=torch.float32)

    # Shuffle and select subset
    total_size = len(X)
    indices = torch.randperm(total_size)
    selected_size = int(total_size * data_proportion)
    selected_indices = indices[:selected_size]

    X = X[selected_indices]
    y = y[selected_indices]

    # Train/val split
    trn_size = int(len(X) * train_size)
    X_train, X_val = X[:trn_size], X[trn_size:]
    y_train, y_val = y[:trn_size], y[trn_size:]

    if batch_size < 1:
        batch_size = X_train.shape[0]

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return (
        train_loader,
        val_loader,
        {"input_dim": obs_dim + action_dim, "output_dim": obs_dim},
    )


def estimate_trace_sigma(train_loader: torch.utils.data.DataLoader) -> float:
    """Estimate trace of the covariance matrix of targets from training data."""
    y_train = []
    for _, yb in train_loader:
        y_train.append(yb)
    y_train = torch.cat(y_train, dim=0)
    y_mean = torch.mean(y_train, dim=0)
    y_centered = y_train - y_mean
    covariance_matrix = torch.cov(y_centered.T)
    trace_sigma = torch.trace(covariance_matrix).item()
    return float(trace_sigma)


def build_policy(
    input_dim: int, output_dim: int, hidden_sizes: List[int], fixed_logstd: bool
) -> Policy.MLPPolicy:
    return Policy.MLPPolicy(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        fixed_logstd=fixed_logstd,
    )


def df_from_metrics(
    metrics: Dict, config: Dict, exp_idx: int, train_or_val: str
) -> pd.DataFrame:
    df = pd.DataFrame(metrics).reset_index().rename(columns={"index": "epoch"})
    for k, v in config.items():
        df[k] = v
    df["experiment"] = exp_idx
    df["train_val"] = train_or_val
    return df


@ray.remote(num_gpus=0.1)
def run_experiment_task(
    *,
    dataset_name: str,
    train_size: float,
    data_proportion: float,
    batch_size: int,
    hidden_sizes: List[int],
    fixed_logstd: bool,
    n_updates: int,
    learning_rate: float,
    use_wandb: bool,
    wandb_project: str,
    env_id: str,
    log_dir: str,
    early_stopping_patience: int,
    exp_idx: int,
    loss_cfg: Dict,
    parquet_dir: str,
) -> Dict:
    """Single experiment worker: builds data, policy, loss; trains; saves Parquet; returns metadata."""
    # Device chosen by Ray via CUDA_VISIBLE_DEVICES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    train_loader, val_loader, data_cfg = generate_data_minari(
        dataset_name=dataset_name,
        train_size=train_size,
        data_proportion=data_proportion,
        batch_size=batch_size,
    )

    # Build policy fresh per run
    policy = build_policy(
        input_dim=data_cfg["input_dim"],
        output_dim=data_cfg["output_dim"],
        hidden_sizes=hidden_sizes,
        fixed_logstd=fixed_logstd,
    )

    # Build loss function from config
    loss_type = loss_cfg["type"]
    if loss_type == "MSE":
        loss_fn: L.LossFunction = L.MSE()
    elif loss_type == "NLL":
        loss_fn = L.NLL()
    elif loss_type == "PG":
        # reward_fn requires matrix U
        if loss_cfg["U_type"] == "I":
            U = torch.eye(data_cfg["output_dim"]).to(device)
            U_label = r"PG($U=I$)"
        elif loss_cfg["U_type"] == "scaled":
            scale_val = float(loss_cfg["U_scale"])
            U = scale_val * torch.eye(data_cfg["output_dim"]).to(device)
            U_label = r"PG($U=\frac{\lambda n}{Tr(\Sigma)}I$)"
        else:
            raise ValueError(f"Unknown U_type: {loss_cfg['U_type']}")

        reward_fn = R.Mahalanobis(matrix=U)
        loss_fn = L.PG(
            reward_fn=reward_fn,
            n_generations=int(loss_cfg["n_generations"]),
            use_rsample=bool(loss_cfg["use_rsample"]),
            reward_transform=str(loss_cfg["reward_transform"]),
            entropy_weight=float(loss_cfg["entropy_weight"]),
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    # Logger per run
    exp_name = f"{loss_fn.name}_{exp_idx}"
    logger, _, ts_writer = setup_logger(
        logger_name="nll_to_po",
        log_dir=log_dir,
        env_id=env_id,
        exp_name=exp_name,
    )

    # Config metadata for DF
    run_cfg = {
        "dataset_name": dataset_name,
        "train_size": train_size,
        "data_proportion": data_proportion,
        "batch_size": batch_size,
        "hidden_sizes": json.dumps(hidden_sizes),
        "fixed_logstd": fixed_logstd,
        "learning_rate": learning_rate,
        "n_updates": n_updates,
        "loss_type": loss_fn.name,
    }
    # Add PG-specific labels (optional columns may be missing for MSE/NLL)
    if loss_type == "PG":
        run_cfg.update(
            {
                "n_generations": int(loss_cfg["n_generations"]),
                "use_rsample": bool(loss_cfg["use_rsample"]),
                "reward_transform": str(loss_cfg["reward_transform"]),
                "entropy_weight": float(loss_cfg["entropy_weight"]),
                "reward_fn": "Mahalanobis",
                "U": U_label,
            }
        )

    # Optionally enable WandB
    wandb_run = None
    if use_wandb:
        import wandb  # local import to avoid hard dep when unused

        wandb_run = wandb.init(project=wandb_project, config=run_cfg)

    # Train
    _, train_metrics, val_metrics = train_single_policy(
        policy=policy,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_function=loss_fn,
        n_updates=n_updates,
        learning_rate=learning_rate,
        wandb_run=wandb_run,
        tensorboard_writer=ts_writer,
        logger=logger,
        device=device,
        early_stopping_patience=early_stopping_patience,
    )

    if use_wandb and wandb_run is not None:
        wandb_run.finish()

    # Build DataFrames
    df_list: List[pd.DataFrame] = []
    df_list.append(df_from_metrics(train_metrics, run_cfg, exp_idx, "train"))
    df_list.append(df_from_metrics(val_metrics, run_cfg, exp_idx, "val"))
    df = pd.concat(df_list, ignore_index=True)

    # Persist parquet (efficient)
    os.makedirs(parquet_dir, exist_ok=True)
    run_id = str(uuid.uuid4())
    parquet_path = os.path.join(parquet_dir, f"results_{run_id}.parquet")
    df.to_parquet(parquet_path, index=False, compression="zstd")

    return {
        "parquet_path": parquet_path,
        "loss_type": loss_fn.name,
        "exp_idx": exp_idx,
        "rows": len(df),
    }


def main(args: ExperimentConfig):
    # Ensure Ray initialized
    ray.init(
        address=args.ray_address, ignore_reinit_error=True, include_dashboard=False
    )

    # Prepare a single data pass to compute trace_sigma and dims for PG scaling
    train_loader_probe, _, data_cfg_probe = generate_data_minari(
        dataset_name=args.dataset,
        train_size=args.train_size,
        data_proportion=args.data_proportion,
        batch_size=args.batch_size,
    )
    trace_sigma = estimate_trace_sigma(train_loader_probe)

    # Build tasks
    tasks = []
    parquet_dir = os.path.abspath(args.results_dir)
    os.makedirs(parquet_dir, exist_ok=True)

    # Helper to enqueue a task
    def enqueue(loss_cfg: Dict, exp_idx: int):
        task = run_experiment_task.remote(
            dataset_name=args.dataset,
            train_size=args.train_size,
            data_proportion=args.data_proportion,
            batch_size=args.batch_size,
            hidden_sizes=args.hidden_sizes,
            fixed_logstd=args.fixed_logstd,
            n_updates=args.n_updates,
            learning_rate=args.learning_rate,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            env_id=args.env_id,
            log_dir=args.log_dir,
            early_stopping_patience=args.early_stopping_patience,
            exp_idx=exp_idx,
            loss_cfg=loss_cfg,
            parquet_dir=parquet_dir,
        )
        tasks.append(task)

    # Enqueue MSE and NLL
    for exp_idx in range(args.n_experiments):
        enqueue({"type": "MSE"}, exp_idx)
    for exp_idx in range(args.n_experiments):
        enqueue({"type": "NLL"}, exp_idx)

    # Enqueue PG variants
    n = data_cfg_probe["output_dim"]
    for exp_idx in range(args.n_experiments):
        for entropy_weight in args.entropy_weights:
            for U_choice in ["I", "scaled"]:
                loss_cfg: Dict = {
                    "type": "PG",
                    "n_generations": args.n_generations,
                    "use_rsample": args.pg_rsample,
                    "reward_transform": args.reward_transform,
                    "entropy_weight": float(entropy_weight),
                    "U_type": U_choice,
                }
                if U_choice == "scaled":
                    # scaled factor = (lambda * n) / trace_sigma
                    U_scale = (float(entropy_weight) * float(n)) / max(
                        trace_sigma, 1e-12
                    )
                    loss_cfg["U_scale"] = U_scale
                enqueue(loss_cfg, exp_idx)

    # Collect results
    results: List[Dict] = ray.get(tasks) if tasks else []
    manifest_path = os.path.join(parquet_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Optionally merge into a single parquet
    if args.merge_results and results:
        frames = []
        for r in results:
            try:
                frames.append(pd.read_parquet(r["parquet_path"]))
                os.remove(r["parquet_path"])  # Clean up individual files after merging
            except Exception as e:
                print(f"Warning: failed to read {r['parquet_path']}: {e}")
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            merged_path = os.path.join(parquet_dir, "results_all.parquet")
            merged.to_parquet(merged_path, index=False, compression="zstd")
            print(f"Saved merged results to {merged_path}")

    print(
        f"Wrote {len(results)} run artifacts to {parquet_dir}. Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args=args)
