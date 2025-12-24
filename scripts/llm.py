import secrets

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import func as torchfunc  # torch.func.functional_call on Torch 2.x
import torchopt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

import nll_to_po.models.reward_network as RM
from nll_to_po.training.utils import (
    set_seed_everywhere,
)
# import nll_to_po.training.loss as L
# import nll_to_po.training.reward as R

seed = secrets.randbits(32)
set_seed_everywhere(seed)
print(f"Seed: {seed}")


# -------------------------
# Utilities: make functional
# -------------------------
def make_functional_with_grads_device(module: nn.Module, device: torch.device):
    """
    Simple helper: returns (module, params_tuple) where params_tuple are leaf tensors with requires_grad True
    Use torch.func.functional_call with dict(zip(names, params_tuple))
    """
    params = tuple(
        p.detach().clone().to(device).requires_grad_(True) for p in module.parameters()
    )
    names = [n for n, _ in module.named_parameters()]
    return module, params, names


def params_tuple_to_dict(names, params_tuple):
    return dict(zip(names, params_tuple))


# -------------------------
# Model selection (small, runnable)
# -------------------------
LM_MODEL_NAME = "Qwen/Qwen3-0.6B"  # change to a larger model if you have resources
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(LM_MODEL_NAME)
lm = AutoModelForCausalLM.from_pretrained(LM_MODEL_NAME, device_map="auto")
print(f"Loaded LM model {LM_MODEL_NAME}")
# lm.eval()

# Reward: use your class
reward_model = RM.BertEmbeddingMahalanobisReward(
    train_encoder=False, train_matrix=True
).to(device)

# Functional wrappers
lm_fn, lm_params, lm_param_names = make_functional_with_grads_device(lm, device)
reward_fn, reward_params, reward_param_names = make_functional_with_grads_device(
    reward_model, device
)


# -------------------------
# Sampling: manual token-by-token (functional)
# -------------------------
@torch.no_grad()
def greedy_logprobs_from_logits(logits, actions):
    # logits: [B, V] ; actions: [B, 1]
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(-1, actions)  # [B, 1]


def sample_batch_policy(policy_params, prompts_input_ids, gen_len=16):
    """
    Token-by-token sampling (multinomial) while computing log-probs for the chosen tokens.
    policy_params: tuple of tensors (functional params)
    prompts_input_ids: LongTensor [B, T_in]
    Returns:
      generated_input_ids: [B, T_in + gen_len]
      actions: [B, gen_len]
      sum_log_probs: [B] (sum across time)
      per_step_logprobs: [B, gen_len]
      new_logits_list: list of logits per step (optionally for KL)
    """
    B = prompts_input_ids.size(0)
    device = prompts_input_ids.device
    generated = prompts_input_ids.clone()
    sum_log_probs = torch.zeros(B, device=device)
    per_step_logprobs = []
    new_logits_list = []
    # old_logprob_per_step = []
    actions_list = []

    for t in range(gen_len):
        # run forward to get logits for current sequence
        params_dict = params_tuple_to_dict(lm_param_names, policy_params)
        outputs = torchfunc.functional_call(
            lm_fn,
            params_dict,
            (),
            {"input_ids": generated, "return_dict": True},
        )
        logits = outputs.logits  # [B, L, V]
        last_logits = logits[:, -1, :]  # [B, V]
        # sample
        probs = F.softmax(last_logits, dim=-1)
        actions = torch.multinomial(probs, num_samples=1)  # [B,1]
        logp = torch.log(torch.gather(probs, 1, actions).clamp(min=1e-12))  # [B,1]
        per_step_logprobs.append(logp)
        sum_log_probs = sum_log_probs + logp.squeeze(-1)
        new_logits_list.append(last_logits)
        actions_list.append(actions)
        generated = torch.cat([generated, actions], dim=1)

    # after building `generated` tensor (B, T_total)
    generated_cpu = generated.detach().cpu()
    generated_texts = [
        tokenizer.decode(g.tolist(), skip_special_tokens=True) for g in generated_cpu
    ]
    per_step_logprobs = torch.cat(per_step_logprobs, dim=1)  # [B, gen_len]
    actions = torch.cat(actions_list, dim=1)  # [B, gen_len]
    return (
        generated,
        actions,
        sum_log_probs,
        per_step_logprobs,
        new_logits_list,
        generated_texts,
    )


# -------------------------
# GRPO components (inner objective)
# -------------------------
def compute_rewards_functional(reward_params_tuple, prompts, generated_texts):
    """
    reward_model is used functionally
    prompts/generations are lists of strings of length B
    """
    params_dict = params_tuple_to_dict(reward_param_names, reward_params_tuple)
    # reward_fn expects (y, y_hat) per your class (list of strings)
    # move reward_model to right device already handled by parameters' devices
    rewards = torchfunc.functional_call(
        reward_fn, params_dict, (prompts, generated_texts), {}
    )
    # ensure shape [B]
    return rewards


def compute_approx_kl(old_logits_list, new_logits_list):
    """
    Estimate KL(P_old || P_new) using sampled tokens' distributions:
      per step: sum_v p_old(v) * (log p_old(v) - log p_new(v))
    We have logits per step for old and new policies. Return mean per-episode KL scalar.
    old_logits_list/new_logits_list: lists of [B,V]
    """
    kl_steps = []
    for old_logits, new_logits in zip(old_logits_list, new_logits_list):
        old_logp = F.log_softmax(old_logits, dim=-1)
        new_logp = F.log_softmax(new_logits, dim=-1)
        p_old = old_logp.exp()
        kl_step = (p_old * (old_logp - new_logp)).sum(dim=-1)  # [B]
        kl_steps.append(kl_step)
    # [B, gen_len]
    kl_per_episode = torch.stack(kl_steps, dim=1).sum(dim=1)
    return kl_per_episode.mean()


# inner objective (policy_params, reward_params, prompts -> scalar loss)
def inner_objective(
    policy_params,
    reward_params,
    prompts_input_ids,
    prompts_texts,
    gen_len=16,
    clip_eps=0.2,
    kl_coef=0.01,
    entropy_coeff=0.0,
):
    """
    Steps:
     - sample responses using current policy (policy_params)
     - compute log_probs under current policy (new_logp)
     - compute baseline and advantages from reward_model (functional)
     - compute likelihood ratio with stored old_logp (we will store old policy rollouts as initial policy before updates)
     - compute clipped surrogate loss, add KL penalty between old and new policies
    """
    # B = prompts_input_ids.size(0)
    # device = prompts_input_ids.device

    # 1) sample under *current* policy (this will produce new_logp)
    (
        generated,
        actions,
        new_sum_log_probs,
        new_per_step_logprobs,
        new_logits_list,
        generated_texts,
    ) = sample_batch_policy(policy_params, prompts_input_ids, gen_len=gen_len)

    # But we need an "old" policy baseline for GRPO. In the simple (and common) variant we treat the policy
    # *before any inner gradient step* as the old policy. For the inner objective during optimization we want
    # to compute grads using the surrogate computed from the gradient step's log-probs.
    #
    # To keep the prototype simple: we'll approximate "old_logprob" by re-evaluating the policy with detached params
    # equal to policy_params but with no gradients (this is equivalent to making the previous policy the same as current).
    # For a proper GRPO update you should store the policy before inner updates; here we compute ratios relative to a
    # detached copy (which will produce ratio=1 initially and meaningful values after actual param updates).

    # params_dict_detached = params_tuple_to_dict(
    #     lm_param_names, tuple(p.detach() for p in policy_params)
    # )

    # compute "old" logits sequence by running with detached params
    # For efficiency, we will run one pass to get last-step logits for each sampling step:
    # (in this prototype we will approximate old logits == new logits at first iteration; still works to exercise gradients)
    old_logits_list = [logit_list.detach() for logit_list in new_logits_list]

    # 2) compute rewards using reward function (functionally)
    # reward model expects (y, y_hat) where y is ground-truth (we'll use prompt continuation as reference or empty)
    # For simple prototype: treat prompts_texts as "y" and generated_texts as "y_hat"
    rewards = compute_rewards_functional(
        reward_params, prompts_texts, generated_texts
    )  # [B]

    # advantage estimation: simple baseline (mean)
    advantages = rewards - rewards.mean()
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )

    # 3) compute ratios: ratio = exp(new_logprob - old_logprob)
    # old_sum_log_probs approximated with detached copy (here new==old initially)
    old_sum_log_probs = new_sum_log_probs.detach()
    ratios = torch.exp(new_sum_log_probs - old_sum_log_probs)  # [B]

    # 4) PPO-style clipped surrogate
    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    surrogate = -torch.mean(torch.min(unclipped, clipped))

    # 5) optional KL penalty estimate between old and new (approx on distributions seen)
    kl_est = compute_approx_kl(old_logits_list, new_logits_list)
    kl_penalty = kl_coef * kl_est

    # 6) entropy regularizer (optional) : approximate with new logits
    entropy = 0.0
    if entropy_coeff > 0.0:
        ent_steps = []
        for logits in new_logits_list:
            p = F.softmax(logits, dim=-1)
            logp = F.log_softmax(logits, dim=-1)
            ent = -(p * logp).sum(dim=-1)  # [B]
            ent_steps.append(ent)
        entropy = torch.stack(ent_steps, dim=1).sum(dim=1).mean()
        entropy_loss = -entropy_coeff * entropy
    else:
        entropy_loss = 0.0

    total_loss = surrogate + kl_penalty + entropy_loss
    return total_loss


# -------------------------
# inner_solver (functional) with torchopt custom_root
# -------------------------
# optimality condition is gradient w.r.t policy params == 0
@torchopt.diff.implicit.custom_root(
    torch.func.grad(inner_objective, argnums=0),  # gradient w.r.t policy params
    argnums=1,  # we want gradients w.r.t reward_params (meta)
    solve=torchopt.linear_solve.solve_normal_cg(maxiter=3, atol=1e-6),
)
def inner_solver(
    policy_params,
    reward_params,
    prompts_input_ids,
    prompts_texts,
    n_inner_steps=3,
    inner_lr=1e-4,
):
    """
    functional inner solver: does a few functional Adam updates of policy_params to minimize inner_objective
    returns optimized policy_params (tuple)
    """
    # create functional Adam optimizer
    inner_opt = torchopt.adam(lr=inner_lr)
    state = inner_opt.init(policy_params)

    for _ in range(n_inner_steps):
        detached_policy_params = tuple(
            p.detach().requires_grad_(True) for p in policy_params
        )
        detached_reward_params = tuple(p.detach() for p in reward_params)
        detached_prompts_input_ids = prompts_input_ids.detach()

        # compute grads wrt policy params for inner_objective
        grads = torch.func.grad(inner_objective, argnums=0)(
            detached_policy_params,
            detached_reward_params,
            detached_prompts_input_ids,
            prompts_texts,
        )
        # clip grads
        grads = [torch.clamp(g, -5.0, 5.0) for g in grads]
        updates, state = inner_opt.update(grads, state, inplace=False)
        updates = tuple(updates)
        policy_params = torchopt.apply_updates(policy_params, updates, inplace=False)
    return tuple(policy_params)


# -------------------------
# outer loss (supervised CE)
# -------------------------
def outer_loss_fn(
    reward_params, init_policy_params, supervised_input_ids, supervised_labels
):
    # 1) adapt policy via inner_solver using RL prompts
    # For the toy example we'll use the supervised prompts also as RL prompts; in practice these differ
    adapted_policy_params = inner_solver(
        init_policy_params,
        reward_params,
        supervised_input_ids,
        [""] * supervised_input_ids.size(0),
    )
    # 2) compute CE loss on supervised set with adapted policy
    params_dict = params_tuple_to_dict(lm_param_names, adapted_policy_params)
    outputs = torchfunc.functional_call(
        lm_fn,
        params_dict,
        (supervised_input_ids,),
        {"input_ids": supervised_input_ids, "return_dict": True},
    )
    logits = outputs.logits  # [B, T, V]
    # cross entropy across tokens (shift labels accordingly); simple flatten CE
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = supervised_labels[:, 1:].contiguous()
    ce_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    return ce_loss


# -------------------------
# tiny toy dataset
# -------------------------
def build_tiny_dataset(tokenizer, device):
    texts = [
        "The capital of France is",
        "The largest mammal is the",
        "The speed of light in vacuum is approximately",
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    labels = input_ids.clone().to(device)  # next-token prediction labels (toy)
    return input_ids, labels, texts


if __name__ == "__main__":
    supervised_input_ids, supervised_labels, supervised_texts = build_tiny_dataset(
        tokenizer, device
    )

    # -------------------------
    # Outer optimization loop (functional outer Adam)
    # -------------------------
    outer_lr = 1e-4
    outer_opt = torchopt.adam(lr=outer_lr)
    outer_state = outer_opt.init(reward_params)

    n_outer_epochs = 3

    for epoch in range(n_outer_epochs):
        # ensure reward_params requires grad
        reward_params = tuple(
            p.detach().clone().requires_grad_(True) for p in reward_params
        )

        # initialize policy params fresh each outer loop
        init_policy_params = tuple(
            p.detach().clone().requires_grad_(True) for p in lm_params
        )

        # compute outer loss
        loss = outer_loss_fn(
            reward_params, init_policy_params, supervised_input_ids, supervised_labels
        )
        print(f"[outer {epoch}] outer_loss = {loss.item():.6f}")

        # grads w.r.t reward params (meta)
        grads = torch.autograd.grad(loss, reward_params, create_graph=False)

        # clip gradients
        grads = [torch.clamp(g, -10.0, 10.0) for g in grads]

        # functional Adam update
        updates, outer_state = outer_opt.update(grads, outer_state, inplace=False)
        updates = tuple(updates)
        new_reward_params = torchopt.apply_updates(
            reward_params, updates, inplace=False
        )

        # copy new params back into reward_model
        for p_module, new_p in zip(reward_model.parameters(), new_reward_params):
            p_module.data.copy_(new_p.data)

        # replace reward_params tuple for next epoch
        reward_params = tuple(
            p.detach().clone().requires_grad_(True) for p in new_reward_params
        )

    print("Done")
