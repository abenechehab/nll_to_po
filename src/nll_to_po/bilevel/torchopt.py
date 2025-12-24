from typing import Optional
from tqdm import tqdm

import torch
import torchopt
from torch.utils.data import DataLoader

import nll_to_po.training.loss as L
import nll_to_po.training.reward as R


# -----------------------
# Helper: make functional params with grad
# -----------------------
def make_functional_with_grads(module, device):
    params = dict(module.named_parameters())
    params = [
        p.detach().clone().requires_grad_(True).to(device) for _, p in params.items()
    ]
    return module.to(device), params


# -----------------------
# Main bilevel routine
# -----------------------
def run_bilevel_optimization_implicit(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    num_classes: int,
    nll_loss_constructor: L.LossFunction,
    pg_loss_constructor: L.LossFunction,
    reward_fn_constructor: R.RewardFunction,
    policy_model_constructor: torch.nn.Module,
    n_outer_epochs=10,
    n_inner_epochs=5,
    inner_lr: float = 1e-3,
    outer_lr: float = 1e-3,
    device: str = "cpu",
    entropy_weight: float = 1.0,
    n_generations: int = 5,
    outer_scheduler_patience: int = 0,
    early_stopping_patience: int = 0,
    reward_network: Optional[torch.nn.Module] = None,
    scheduler_factor: float = 0.75,
    scheduler_min_lr: float = 1e-6,
    policy_base_state: Optional[dict] = None,
):
    # reward
    reward_network = reward_network.to(device)
    reward_model, reward_params = make_functional_with_grads(reward_network, device)

    # reward loss: nll_loss
    nll_loss = nll_loss_constructor()

    # Tracking
    train_accuracy = []
    val_accuracy = []
    outer_losses = []
    val_outer_losses = []
    inner_losses = []
    inner_rewards = []
    # Initialize tracking dictionaries for reward network parameters
    U_coeff = {}
    U_coeff_grad = {}
    # Get parameter names and shapes from reward network
    for name, param in reward_network.named_parameters():
        if param.requires_grad is True:
            if param.dim() == 2:  # Matrix parameter
                rows, cols = param.shape
                for i in range(rows):
                    for j in range(cols):
                        key = f"{name}_{i + 1}_{j + 1}"
                        U_coeff[key] = []
                        U_coeff_grad[key] = []
            elif param.dim() == 1:  # Vector parameter
                for i in range(param.shape[0]):
                    key = f"{name}_{i + 1}"
                    U_coeff[key] = []
                    U_coeff_grad[key] = []
            else:  # Scalar parameter
                U_coeff[name] = []
                U_coeff_grad[name] = []

    # Inner-loop objective function
    # The optimality function: grad(inner_objective)
    def inner_objective(policy_params, reward_params, X, y, policy_model, reward_model):
        reward_params_dict = dict(
            zip(dict(reward_model.named_parameters()).keys(), reward_params)
        )
        reward_fn = reward_fn_constructor(
            reward_model=reward_model,
            reward_params=reward_params_dict,
            num_classes=num_classes,
        )

        # policy loss
        pg_loss = pg_loss_constructor(
            reward_fn=reward_fn,
            n_generations=n_generations,
            use_rsample=False,
            reward_transform="none",
            entropy_weight=entropy_weight,
        )

        # Reconstruct params dict (for functional_call)
        policy_params_dict = dict(
            zip(dict(policy_model.named_parameters()).keys(), policy_params)
        )
        loss, _ = pg_loss.compute_loss(
            policy_model=policy_model,
            policy_params=policy_params_dict,
            X=X,
            y=y,
            reward_model=reward_model,
            reward_params=reward_params_dict,
        )
        return loss

    # Optimality Condition is: the gradient w.r.t inner-loop optimal params is 0 (we achieve so by
    # specifying argnums=0 in functorch.grad) the argnums=1 specify which meta-parameter we want to
    # backpropogate, in this case we want to backpropogate to the initial parameters so we set it as 1.
    # You can also set argnums as (1, 2) if you want to backpropogate through multiple meta-parameters

    # Here we pass argnums=1 to the custom_root. That means we want to compute the gradient of
    # optimal_params w.r.t. the 1-indexed argument in inner_solver, i.e., params.
    # torchopt.linear_solve.solve_normal_cg specify that we use the conjugate gradient based linear solver
    @torchopt.diff.implicit.custom_root(
        torch.func.grad(inner_objective, argnums=0),  # optimality function
        argnums=1,
        solve=torchopt.linear_solve.solve_normal_cg(maxiter=1, atol=1e-6),
        # solve=torchopt.linear_solve.solve_inv(ns=True, maxiter=100, alpha=0.1),
    )
    def inner_solver(policy_params, reward_params, X, y, policy_model, reward_model):
        reward_params_dict = dict(
            zip(dict(reward_model.named_parameters()).keys(), reward_params)
        )
        reward_fn = R.FuncOneHotRewardNetwork(
            reward_model=reward_model,
            reward_params=reward_params_dict,
            num_classes=num_classes,
        )

        # policy loss
        pg_loss = pg_loss_constructor(
            reward_fn=reward_fn,
            n_generations=n_generations,
            use_rsample=False,
            reward_transform="none",
            entropy_weight=entropy_weight,
        )

        def inner_loss_fn(policy_params, X, y):
            # Reconstruct params dict (for functional_call)
            policy_params_dict = dict(
                zip(dict(policy_model.named_parameters()).keys(), policy_params)
            )
            loss, _ = pg_loss.compute_loss(
                policy_model=policy_model,
                policy_params=policy_params_dict,
                X=X,
                y=y,
                reward_model=reward_model,
                reward_params=reward_params_dict,
            )
            return loss

        # Create differentiable SGD optimizer
        inner_optimizer = torchopt.adam(lr=inner_lr)  # , momentum=0.9)  # , eps=1e-8)
        inner_opt_state = inner_optimizer.init(policy_params)

        # Inner loop with torchopt
        inner_losses_per_iter = []
        inner_rewards_per_iter = []
        for _ in range(n_inner_epochs):
            # Compute gradients and update
            grads = torch.func.grad(inner_loss_fn)(
                policy_params,
                X,
                y,
            )
            # Clip gradients to prevent explosion
            grads = [torch.clamp(g, -1.0, 1.0) for g in grads]
            updates, inner_opt_state = inner_optimizer.update(
                grads, inner_opt_state, inplace=False
            )
            policy_params = torchopt.apply_updates(
                policy_params, updates, inplace=False
            )

            # Track loss
            with torch.no_grad():
                # Reconstruct params dict (for functional_call)
                policy_params_dict = dict(
                    zip(dict(policy_model.named_parameters()).keys(), policy_params)
                )
                inner_loss, metrics = pg_loss.compute_loss(
                    policy_model,
                    policy_params_dict,
                    X,
                    y,
                    reward_model=reward_model,
                    reward_params=reward_params_dict,
                )
                inner_losses_per_iter.append(float(inner_loss))
                inner_rewards_per_iter.append(metrics["reward_mean"])
        inner_losses.append(inner_losses_per_iter)
        inner_rewards.append(inner_rewards_per_iter)

        return tuple(policy_params)

    def outer_loss_fn(
        reward_params, policy_params, X, y, X_val, y_val, policy_model, reward_model
    ):
        policy_params = inner_solver(
            policy_params, reward_params, X, y, policy_model, reward_model
        )

        policy_params_dict = dict(
            zip(dict(policy_model.named_parameters()).keys(), policy_params)
        )

        # outer loss
        loss, train_metrics = nll_loss.compute_loss(
            policy_model=policy_model,
            policy_params=policy_params_dict,
            X=X,
            y=y,
        )

        # Validation loss
        with torch.no_grad():
            val_loss, val_metrics = nll_loss.compute_loss(
                policy_model=policy_model,
                policy_params=policy_params_dict,
                X=X_val,
                y=y_val,
            )
        return loss, val_loss, train_metrics["accuracy"], val_metrics["accuracy"]

    s_patience_counter = 0
    current_outer_lr = outer_lr

    es_patience_counter = 0
    best_val_loss = float("inf")
    best_reward_params = None

    for outer_epoch in tqdm(range(n_outer_epochs), desc="Outer Epochs"):
        # -----------------------
        # Outer loop: update reward params
        # -----------------------

        # Reset reward params gradient
        reward_params = [p.detach().clone().requires_grad_(True) for p in reward_params]

        # reward optimizer with current learning rate
        outer_optimizer = torchopt.adam(lr=current_outer_lr)
        outer_opt_state = outer_optimizer.init(reward_params)

        # Reset policy each outer iteration
        policy = policy_model_constructor(
            input_dim=input_dim, output_dim=num_classes
        ).to(device)
        if policy_base_state:
            policy.load_state_dict({k: v.clone() for k, v in policy_base_state.items()})
        policy_model, policy_params = make_functional_with_grads(policy, device)

        # Process batches for outer loop
        epoch_outer_losses = []
        epoch_val_losses = []
        epoch_train_accuracies = []
        epoch_val_accuracies = []

        accumulated_grads = None
        num_batches = 0

        # Create iterators for train and validation loaders
        val_iter = iter(val_loader)

        for _, (X_train_batch, y_train_batch, *_) in enumerate(train_loader):
            # Get validation batch (cycle through val_loader if needed)
            try:
                X_val_batch, y_val_batch, *_ = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                X_val_batch, y_val_batch, *_ = next(val_iter)

            # Compute gradients and losses for this batch
            loss, val_loss, train_acc, val_acc = outer_loss_fn(
                reward_params,
                policy_params,
                X_train_batch.to(device),
                y_train_batch.to(device),
                X_val_batch.to(device),
                y_val_batch.to(device),
                policy_model,
                reward_model,
            )

            epoch_outer_losses.append(float(loss))
            epoch_val_losses.append(float(val_loss))
            epoch_train_accuracies.append(float(train_acc))
            epoch_val_accuracies.append(float(val_acc))

            grads = torch.autograd.grad(loss, reward_params, create_graph=True)
            # Clip gradients to prevent explosion
            grads = [torch.clamp(g, -10.0, 10.0) for g in grads]

            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = [g.clone() for g in grads]
            else:
                for i, g in enumerate(grads):
                    accumulated_grads[i] += g

            num_batches += 1

        # Average accumulated gradients
        accumulated_grads = [g / num_batches for g in accumulated_grads]

        # Update parameters with averaged gradients
        updates, outer_opt_state = outer_optimizer.update(
            accumulated_grads, outer_opt_state, inplace=False
        )
        reward_params = torchopt.apply_updates(reward_params, updates)

        # Store epoch averages
        avg_outer_loss = sum(epoch_outer_losses) / len(epoch_outer_losses)
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        avg_train_acc = sum(epoch_train_accuracies) / len(epoch_train_accuracies)
        avg_val_acc = sum(epoch_val_accuracies) / len(epoch_val_accuracies)

        outer_losses.append(avg_outer_loss)
        val_outer_losses.append(avg_val_loss)
        train_accuracy.append(avg_train_acc)
        val_accuracy.append(avg_val_acc)

        # Early stopping: check if current validation loss is the best
        if early_stopping_patience:
            current_val_loss = avg_val_loss
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_reward_params = [p.detach().clone() for p in reward_params]
                es_patience_counter = 0
            else:
                es_patience_counter += 1
                reward_params = [p.detach().clone() for p in best_reward_params]
                if es_patience_counter >= early_stopping_patience:
                    print(f"Early stopping at outer epoch {outer_epoch}")
                    break

        if outer_scheduler_patience:
            current_val_loss = avg_val_loss
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                s_patience_counter = 0
            else:
                s_patience_counter += 1
            if (
                s_patience_counter >= outer_scheduler_patience
                and current_outer_lr > scheduler_min_lr
            ):
                current_outer_lr = max(
                    current_outer_lr * scheduler_factor, scheduler_min_lr
                )

        # Track all reward network parameters
        param_idx = 0
        for name, param in reward_network.named_parameters():
            if param.requires_grad is True:
                if param.dim() == 2:  # Matrix parameter
                    rows, cols = param.shape
                    for i in range(rows):
                        for j in range(cols):
                            key = f"{name}_{i + 1}_{j + 1}"
                            U_coeff[key].append(reward_params[param_idx][i, j].item())
                            U_coeff_grad[key].append(
                                accumulated_grads[param_idx][i, j].item()
                            )
                elif param.dim() == 1:  # Vector parameter
                    for i in range(param.shape[0]):
                        key = f"{name}_{i + 1}"
                        U_coeff[key].append(reward_params[param_idx][i].item())
                        U_coeff_grad[key].append(accumulated_grads[param_idx][i].item())
                else:  # Scalar parameter
                    U_coeff[name].append(reward_params[param_idx].item())
                    U_coeff_grad[name].append(accumulated_grads[param_idx].item())
            param_idx += 1

    return (
        train_accuracy,
        val_accuracy,
        outer_losses,
        val_outer_losses,
        inner_losses,
        inner_rewards,
        U_coeff,
        U_coeff_grad,
    )
