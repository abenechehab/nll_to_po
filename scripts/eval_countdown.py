#!/usr/bin/env python3
"""
Evaluate LoRA checkpoints on the Countdown task (single or batch).

Single checkpoint:
    accelerate launch --num_processes 3 --multi_gpu scripts/eval_countdown.py \
        --model-name Qwen/Qwen3-0.6B \
        --adapter-path logs/.../checkpoint-500 \
        --batch-size 256

All checkpoints in a directory:
    accelerate launch --num_processes 3 --multi_gpu scripts/eval_countdown.py \
        --model-name Qwen/Qwen3-0.6B \
        --adapter-base-path logs/.../run-dir \
        --checkpoint-step 100 \
        --start-checkpoint 100 \
        --end-checkpoint 1000 \
        --batch-size 256

With pre-adapters (merged before fine-tuning):
    accelerate launch --num_processes 3 --multi_gpu scripts/eval_countdown.py \
        --model-name Qwen/Qwen3-0.6B \
        --adapter-base-path logs/.../run-dir \
        --pre-adapters path/to/adapter1 path/to/adapter2 \
        --batch-size 256
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import tyro

from nll_to_po.training.reward import equation_reward_func


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EvalCountdownConfig:
    """Evaluate one or more LoRA checkpoints on the Countdown task."""

    # ---- model ----
    model_name: str
    """Base model name (e.g. Qwen/Qwen3-0.6B)."""

    # ---- single-checkpoint mode ----
    adapter_path: Optional[str] = None
    """Path to a single PEFT adapter. Mutually exclusive with adapter_base_path."""

    # ---- multi-checkpoint mode ----
    adapter_base_path: Optional[str] = None
    """Directory containing checkpoint-N subdirs. All matching checkpoints are
    evaluated sequentially (model + pre-adapters loaded once)."""

    start_checkpoint: Optional[int] = None
    """Minimum checkpoint number to evaluate."""

    end_checkpoint: Optional[int] = None
    """Maximum checkpoint number to evaluate."""

    checkpoint_step: Optional[int] = None
    """Only evaluate checkpoints whose number is divisible by this."""

    checkpoints: Optional[str] = None
    """Comma-separated specific checkpoint numbers (e.g. '100,300,500')."""

    # ---- pre-adapters ----
    pre_adapters: list[str] = field(default_factory=list)
    """Ordered list of adapter paths to load & merge into the base model
    *before* applying each checkpoint adapter.  These should mirror the
    adapters that were merged prior to fine-tuning."""

    # ---- generation / eval ----
    batch_size: int = 256
    """Batch size per GPU."""

    max_samples: Optional[int] = None
    """Cap the number of evaluation samples (useful for debugging)."""

    max_new_tokens: int = 512
    """Maximum tokens to generate per sample."""

    verbose: int = 1
    """Verbosity level (0 = silent, 1 = summary + samples)."""

    # ---- output ----
    save_results: Optional[str] = None
    """JSON file to append per-checkpoint results to."""


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

_CKPT_RE = re.compile(r"checkpoint-(\d+)")


def find_checkpoints(
    base: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: Optional[int] = None,
    specific: Optional[list[int]] = None,
) -> list[Path]:
    base_path = Path(base)
    if not base_path.exists():
        raise ValueError(f"Adapter base path does not exist: {base}")

    hits: list[tuple[int, Path]] = []
    for item in base_path.iterdir():
        if not item.is_dir():
            continue
        m = _CKPT_RE.match(item.name)
        if not m:
            continue
        num = int(m.group(1))

        if specific is not None:
            if num in specific:
                hits.append((num, item))
            continue

        if start is not None and num < start:
            continue
        if end is not None and num > end:
            continue
        if step is not None and num % step != 0:
            continue
        hits.append((num, item))

    hits.sort(key=lambda x: x[0])
    return [p for _, p in hits]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def prepare_prompts(dataset, tokenizer):
    prompts, targets, numbers = [], [], []
    for ex in dataset:
        chat = [
            {"role": "system", "content": ex["prompt"][0]["content"]},
            {"role": "user", "content": ex["prompt"][1]["content"]},
            {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
        ]
        prompts.append(
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        )
        targets.append(ex["target"])
        numbers.append(ex["nums"])
    return prompts, targets, numbers


def generate_batch(model, tokenizer, prompts, *, max_new_tokens: int = 512):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
        return_attention_mask=True,
    ).to(model.device)

    with torch.no_grad():
        underlying = model.module if hasattr(model, "module") else model
        outputs = underlying.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    clean = []
    for comp, prompt in zip(completions, prompts):
        if comp.startswith(prompt):
            clean.append(comp[len(prompt):])
        else:
            clean.append(comp.split("<think>")[-1] if "<think>" in comp else comp)
    return clean


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base_model_with_pre_adapters(
    model_name: str,
    pre_adapters: list[str],
    device: torch.device,
    is_main: bool,
) -> AutoModelForCausalLM:
    """Load the base model then sequentially load & merge each pre-adapter."""
    if is_main:
        print(f"Loading base model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )

    for i, adapter_path in enumerate(pre_adapters, 1):
        if is_main:
            print(f"  Merging pre-adapter {i}/{len(pre_adapters)}: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    return model


def apply_checkpoint_adapter(
    base_model: AutoModelForCausalLM,
    adapter_path: str,
    is_main: bool,
) -> AutoModelForCausalLM:
    """Load a checkpoint adapter on top of the (already-merged) base model,
    merge, and return the plain model."""
    if is_main:
        print(f"  Loading checkpoint adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    return model


# ---------------------------------------------------------------------------
# Single-checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
    cfg: EvalCountdownConfig,
    all_prompts: list[str],
    all_targets: list,
    all_numbers: list,
    adapter_label: str,
) -> dict | None:
    """Run generation + scoring and return metrics dict (main process only)."""

    with accelerator.split_between_processes(
        list(zip(all_prompts, all_targets, all_numbers))
    ) as split_data:
        proc_prompts = [d[0] for d in split_data]
        proc_targets = [d[1] for d in split_data]
        proc_numbers = [d[2] for d in split_data]

    if accelerator.is_main_process:
        print(f"  Generating ({len(proc_prompts)} examples on this process) ...")

    proc_completions: list[str] = []
    for i in tqdm(
        range(0, len(proc_prompts), cfg.batch_size),
        desc=f"GPU {accelerator.process_index}",
        disable=not accelerator.is_main_process,
    ):
        batch = proc_prompts[i : i + cfg.batch_size]
        proc_completions.extend(
            generate_batch(model, tokenizer, batch, max_new_tokens=cfg.max_new_tokens)
        )

    gathered_completions = accelerator.gather_for_metrics(proc_completions)
    gathered_targets = accelerator.gather_for_metrics(proc_targets)
    gathered_numbers = accelerator.gather_for_metrics(proc_numbers)

    if not accelerator.is_main_process:
        return None

    # --- scoring (main process only) ---
    rewards = equation_reward_func(
        gathered_completions, gathered_targets, gathered_numbers, verbose=cfg.verbose
    )
    accuracy = float(np.mean(rewards))
    num_correct = int(sum(rewards))
    total = len(rewards)
    has_tag = sum(1 for c in gathered_completions if "<answer>" in c and "</answer>" in c)
    tag_rate = has_tag / total

    print(f"\n{'=' * 80}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 80}")
    print(f"Model: {cfg.model_name}")
    print(f"Adapter: {adapter_label}")
    if cfg.pre_adapters:
        print(f"Pre-adapters: {cfg.pre_adapters}")
    print(f"Total examples: {total}")
    print(f"Accuracy: {accuracy:.2%} ({num_correct}/{total})")
    print(f"Answer tag format rate: {tag_rate:.2%}")
    print(f"{'=' * 80}")

    result = {
        "config": {
            "model_name": cfg.model_name,
            "adapter": adapter_label,
            "pre_adapters": cfg.pre_adapters,
            "num_gpus": accelerator.num_processes,
            "batch_size_per_gpu": cfg.batch_size,
            "effective_batch_size": cfg.batch_size * accelerator.num_processes,
        },
        "metrics": {
            "accuracy": accuracy,
            "num_correct": num_correct,
            "total": total,
            "answer_tag_rate": tag_rate,
        },
        "examples": [
            {
                "completion": comp,
                "target": tgt,
                "numbers": nums,
                "correct": bool(r == 1.0),
            }
            for comp, tgt, nums, r in zip(
                gathered_completions[:5],
                gathered_targets[:5],
                gathered_numbers[:5],
                rewards[:5],
            )
        ],
    }

    if cfg.verbose > 0:
        print(f"\n{'=' * 80}")
        print("SAMPLE PREDICTIONS")
        print(f"{'=' * 80}")
        for i in range(min(5, total)):
            print(f"\n--- Example {i + 1} ---")
            print(f"Numbers: {gathered_numbers[i]}, Target: {gathered_targets[i]}")
            print(f"Completion: {gathered_completions[i][:200]}...")
            print(f"Correct: {rewards[i] == 1.0}")

    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def append_result(path: str, result: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data: list = []
    if p.exists():
        with open(p) as f:
            data = json.load(f)
    data.append(result)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results appended to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = tyro.cli(EvalCountdownConfig)

    # Validate: exactly one of adapter_path / adapter_base_path (or neither for base model)
    if cfg.adapter_path and cfg.adapter_base_path:
        raise ValueError("Provide --adapter-path (single) or --adapter-base-path (multi), not both.")

    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    # --- resolve checkpoint list ---
    if cfg.adapter_base_path:
        specific = (
            [int(x.strip()) for x in cfg.checkpoints.split(",")]
            if cfg.checkpoints
            else None
        )
        adapter_paths = find_checkpoints(
            cfg.adapter_base_path,
            start=cfg.start_checkpoint,
            end=cfg.end_checkpoint,
            step=cfg.checkpoint_step,
            specific=specific,
        )
        if not adapter_paths:
            if is_main:
                print("No checkpoints found matching the criteria.")
            return
        if is_main:
            print(f"Found {len(adapter_paths)} checkpoints to evaluate:")
            for cp in adapter_paths:
                print(f"  - {cp.name}")
    elif cfg.adapter_path:
        adapter_paths = [Path(cfg.adapter_path)]
    else:
        adapter_paths = []  # base-model-only evaluation

    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- dataset ---
    if is_main:
        print("Loading Countdown-Task-GOLD dataset...")
    dataset = load_dataset("HuggingFaceTB/Countdown-Task-GOLD", "test")["test"]
    if cfg.max_samples is not None:
        dataset = dataset.select(range(min(cfg.max_samples, len(dataset))))
    all_prompts, all_targets, all_numbers = prepare_prompts(dataset, tokenizer)
    if is_main:
        print(f"Dataset size: {len(all_prompts)}")

    # --- base model (+ pre-adapters, loaded once) ---
    base_model = load_base_model_with_pre_adapters(
        cfg.model_name, cfg.pre_adapters, accelerator.device, is_main,
    )

    # --- evaluate ---
    if not adapter_paths:
        # Pure base model evaluation
        base_model.eval()
        result = evaluate_model(
            base_model, tokenizer, accelerator, cfg,
            all_prompts, all_targets, all_numbers,
            adapter_label="None (base model)",
        )
        if result and cfg.save_results:
            append_result(cfg.save_results, result)
    else:
        for i, cp_path in enumerate(adapter_paths, 1):
            if is_main:
                print(f"\n[{i}/{len(adapter_paths)}] {cp_path.name}")

            model = apply_checkpoint_adapter(base_model, str(cp_path), is_main)
            model.eval()

            result = evaluate_model(
                model, tokenizer, accelerator, cfg,
                all_prompts, all_targets, all_numbers,
                adapter_label=str(cp_path),
            )
            if result and cfg.save_results:
                append_result(cfg.save_results, result)

            # Free the merged model to reclaim memory before next checkpoint
            del model
            torch.cuda.empty_cache()

    if is_main:
        print("\nAll evaluations complete!")


if __name__ == "__main__":
    main()