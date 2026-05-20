"""
Distributed evaluation of fine-tuned LLMs on the PubMedQA test set.

Supports evaluating a single adapter or sweeping over multiple checkpoints
found under a common training run directory, with optional pre-adapters
that are loaded and merged once before iterating over checkpoints.

Usage (single adapter):
    accelerate launch --num_processes 3 --multi_gpu eval_pubmed_accelerate.py \
        --model-name Qwen/Qwen3-0.6B \
        --adapter-path ./logs/.../checkpoint-200 \
        --batch-size 64

Usage (all checkpoints with pre-adapters):
    accelerate launch --num_processes 3 --multi_gpu eval_pubmed_accelerate.py \
        --model-name Qwen/Qwen3-0.6B \
        --adapter-base-path ./logs/.../[peft][v15]trl-sft-... \
        --pre-adapters path/to/sft-adapter path/to/grpo-adapter \
        --checkpoint-step 20 \
        --batch-size 64

Standalone (no accelerate, single GPU):
    python eval_pubmed_accelerate.py --model-name Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import field
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

# ── local imports ────────────────────────────────────────────────────────────
from nll_to_po.medical.utils import (
    SYSTEM_PROMPT,
    extract_answer,
    extract_long_answer,
    format_prompt,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI config
# ═══════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class EvalConfig:
    """PubMedQA evaluation configuration."""

    # ── model ────────────────────────────────────────────────────────────────
    model_name: str
    """HuggingFace model id (e.g. Qwen/Qwen3-0.6B)."""

    adapter_path: Optional[str] = None
    """Path to a single PEFT/LoRA adapter directory. Mutually exclusive with
    ``adapter_base_path``; if both are None the base model is evaluated."""

    # ── multi-checkpoint sweep ───────────────────────────────────────────────
    adapter_base_path: Optional[str] = None
    """Root directory of a training run that contains ``checkpoint-*`` sub-dirs.
    When provided, every matching checkpoint is evaluated sequentially."""

    start_checkpoint: Optional[int] = None
    """Minimum checkpoint step to include (inclusive)."""

    end_checkpoint: Optional[int] = None
    """Maximum checkpoint step to include (inclusive)."""

    checkpoint_step: Optional[int] = None
    """Only include checkpoints whose step number is divisible by this value."""

    checkpoints: Optional[str] = None
    """Comma-separated list of specific checkpoint step numbers
    (e.g. '20,40,100'). Overrides start/end/step filters."""

    # ── pre-adapters ─────────────────────────────────────────────────────────
    pre_adapters: list[str] = field(default_factory=list)
    """Ordered list of adapter paths to load & merge into the base model
    *before* applying each checkpoint adapter.  These should mirror the
    adapters that were merged prior to fine-tuning."""

    # ── generation ───────────────────────────────────────────────────────────
    batch_size: int = 64
    """Batch size **per GPU**."""

    max_new_tokens: int = 512
    """Maximum number of new tokens to generate per sample."""

    max_samples: Optional[int] = None
    """Cap the number of test samples (useful for quick debugging)."""

    # ── output ───────────────────────────────────────────────────────────────
    save_results: Optional[str] = "results/results_pubmed.json"
    """Path to a JSON file where results are appended."""

    verbose: int = 0
    """Verbosity level (0 = summary only, >=1 = show sample predictions)."""


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint discovery
# ═══════════════════════════════════════════════════════════════════════════════

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def find_checkpoints(cfg: EvalConfig) -> list[Path]:
    """Return sorted list of checkpoint paths matching the filter criteria."""
    base = Path(cfg.adapter_base_path)
    if not base.exists():
        raise FileNotFoundError(f"adapter_base_path does not exist: {base}")

    specific: set[int] | None = None
    if cfg.checkpoints is not None:
        specific = {int(x.strip()) for x in cfg.checkpoints.split(",")}

    matched: list[tuple[int, Path]] = []
    for item in base.iterdir():
        if not item.is_dir():
            continue
        m = _CHECKPOINT_RE.match(item.name)
        if m is None:
            continue
        step = int(m.group(1))

        if specific is not None:
            if step in specific:
                matched.append((step, item))
            continue

        if cfg.start_checkpoint is not None and step < cfg.start_checkpoint:
            continue
        if cfg.end_checkpoint is not None and step > cfg.end_checkpoint:
            continue
        if cfg.checkpoint_step is not None and step % cfg.checkpoint_step != 0:
            continue
        matched.append((step, item))

    matched.sort(key=lambda t: t[0])
    return [p for _, p in matched]


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_base_model_with_pre_adapters(
    model_name: str,
    pre_adapters: list[str],
    device: torch.device,
    is_main: bool,
) -> AutoModelForCausalLM:
    """Load the base model, then sequentially load & merge each pre-adapter."""
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
    """Load a checkpoint adapter on top of the (already-merged) base, merge,
    and return the plain model."""
    if is_main:
        print(f"  Loading checkpoint adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt preparation
# ═══════════════════════════════════════════════════════════════════════════════


def prepare_prompts(dataset, tokenizer) -> tuple[list[str], list[str], list[str]]:
    """Build tokenizer-formatted prompts and extract ground-truth labels."""
    prompts: list[str] = []
    answers: list[str] = []
    long_answers: list[str] = []

    for sample in dataset:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_prompt(sample)},
        ]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt_str)
        answers.append(sample["final_decision"])
        long_answers.append(sample["LONG_ANSWER"])

    return prompts, answers, long_answers


# ═══════════════════════════════════════════════════════════════════════════════
# Batch generation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
) -> list[str]:
    """Run greedy generation on a batch and return decoded completions only."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
        return_attention_mask=True,
    ).to(model.device)

    with torch.no_grad():
        gen_model = model.module if hasattr(model, "module") else model
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, prompt_len:]
    completions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return completions


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════════


def score_completions(
    completions: list[str],
    answers: list[str],
    verbose: int = 0,
) -> dict:
    """Compute answer-correctness metrics from raw completion strings."""
    rewards: list[float] = []
    for comp, gt in zip(completions, answers):
        predicted = extract_answer(comp)
        correct = predicted is not None and predicted == gt.strip().lower()
        rewards.append(1.0 if correct else 0.0)

    total = len(rewards)
    num_correct = int(sum(rewards))
    accuracy = np.mean(rewards) if total > 0 else 0.0

    has_answer_tag = sum(1 for c in completions if "<answer>" in c and "</answer>" in c)
    has_long_answer_tag = sum(
        1 for c in completions if "<long_answer>" in c and "</long_answer>" in c
    )
    answer_tag_rate = has_answer_tag / total if total else 0.0
    long_answer_tag_rate = has_long_answer_tag / total if total else 0.0

    format_compliance = sum(
        1
        for c in completions
        if extract_answer(c) is not None and extract_long_answer(c) is not None
    )
    format_compliance_rate = format_compliance / total if total else 0.0

    return {
        "rewards": rewards,
        "accuracy": float(accuracy),
        "num_correct": num_correct,
        "total": total,
        "answer_tag_rate": float(answer_tag_rate),
        "long_answer_tag_rate": float(long_answer_tag_rate),
        "format_compliance_rate": float(format_compliance_rate),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Single-model evaluation (runs generation + scoring on a ready model)
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
    cfg: EvalConfig,
    all_prompts: list[str],
    all_answers: list[str],
    all_long_answers: list[str],
    adapter_label: str,
) -> dict | None:
    """Run generation + scoring. Returns result dict on main process, None on workers."""

    with accelerator.split_between_processes(
        list(zip(all_prompts, all_answers, all_long_answers))
    ) as split_data:
        proc_prompts = [d[0] for d in split_data]
        proc_answers = [d[1] for d in split_data]
        proc_long_answers = [d[2] for d in split_data]

    if accelerator.is_main_process:
        print(f"  Generating ({len(proc_prompts)} examples on this process) …")

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
    gathered_answers = accelerator.gather_for_metrics(proc_answers)
    _ = accelerator.gather_for_metrics(proc_long_answers)

    if not accelerator.is_main_process:
        return None

    # ── score ────────────────────────────────────────────────────────────
    metrics = score_completions(gathered_completions, gathered_answers, verbose=cfg.verbose)

    print(f"\n{'=' * 80}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 80}")
    print(f"  Model           : {cfg.model_name}")
    print(f"  Adapter         : {adapter_label}")
    if cfg.pre_adapters:
        print(f"  Pre-adapters    : {cfg.pre_adapters}")
    print(f"  Total examples  : {metrics['total']}")
    print(f"  Accuracy        : {metrics['accuracy']:.2%} ({metrics['num_correct']}/{metrics['total']})")
    print(f"  <answer> rate   : {metrics['answer_tag_rate']:.2%}")
    print(f"  <long_answer>   : {metrics['long_answer_tag_rate']:.2%}")
    print(f"  Format compliant: {metrics['format_compliance_rate']:.2%}")
    print(f"{'=' * 80}")

    if cfg.verbose >= 1:
        n_show = min(5, len(gathered_completions))
        print("\nSAMPLE PREDICTIONS")
        print("-" * 80)
        for i in range(n_show):
            print(f"\n--- Example {i + 1} ---")
            print(f"  Ground truth : {gathered_answers[i]}")
            print(f"  Predicted    : {extract_answer(gathered_completions[i])}")
            pred_long = extract_long_answer(gathered_completions[i])
            print(
                f"  Long answer  : {pred_long[:120] + '…' if pred_long and len(pred_long) > 120 else pred_long}"
            )
            print(f"  Correct      : {metrics['rewards'][i] == 1.0}")

    result_record = {
        "config": {
            "model_name": cfg.model_name,
            "adapter_path": adapter_label,
            "pre_adapters": cfg.pre_adapters,
            "num_gpus": accelerator.num_processes,
            "batch_size_per_gpu": cfg.batch_size,
            "effective_batch_size": cfg.batch_size * accelerator.num_processes,
            "max_new_tokens": cfg.max_new_tokens,
        },
        "metrics": {k: v for k, v in metrics.items() if k != "rewards"},
        "examples": [
            {
                "completion": comp[:500],
                "ground_truth": gt,
                "predicted": extract_answer(comp),
                "correct": bool(r == 1.0),
            }
            for comp, gt, r in zip(
                gathered_completions[:5],
                gathered_answers[:5],
                metrics["rewards"][:5],
            )
        ],
    }

    if cfg.save_results:
        results_path = Path(cfg.save_results)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        data: list = []
        if results_path.exists():
            with open(results_path, "r") as f:
                data = json.load(f)
        data.append(result_record)
        with open(results_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults appended to {cfg.save_results}")

    return result_record


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    cfg = tyro.cli(EvalConfig)
    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    if cfg.adapter_path and cfg.adapter_base_path:
        raise ValueError("Provide --adapter-path (single) or --adapter-base-path (multi), not both.")

    # ── resolve checkpoint list ──────────────────────────────────────────
    if cfg.adapter_base_path is not None:
        adapter_paths = find_checkpoints(cfg)
        if not adapter_paths:
            if is_main:
                print("No checkpoints found matching the given criteria.")
            return
        if is_main:
            print(f"Found {len(adapter_paths)} checkpoint(s) to evaluate:")
            for cp in adapter_paths:
                print(f"  • {cp.name}")
    elif cfg.adapter_path is not None:
        adapter_paths = [Path(cfg.adapter_path)]
    else:
        adapter_paths = []  # base model only

    # ── tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── dataset (loaded once) ────────────────────────────────────────────
    if is_main:
        print("Loading PubMedQA test split …")
    dataset = load_dataset("bigbio/pubmed_qa", trust_remote_code=True, split="validation")
    if cfg.max_samples is not None:
        dataset = dataset.select(range(min(cfg.max_samples, len(dataset))))
    all_prompts, all_answers, all_long_answers = prepare_prompts(dataset, tokenizer)
    if is_main:
        print(f"  Test set size: {len(all_prompts)}")

    # ── base model + pre-adapters (loaded once) ─────────────────────────
    base_model = load_base_model_with_pre_adapters(
        cfg.model_name, cfg.pre_adapters, accelerator.device, is_main,
    )

    # ── evaluate ─────────────────────────────────────────────────────────
    if not adapter_paths:
        base_model.eval()
        evaluate_model(
            base_model, tokenizer, accelerator, cfg,
            all_prompts, all_answers, all_long_answers,
            adapter_label="None (base model)",
        )
    else:
        for i, cp_path in enumerate(adapter_paths, 1):
            if is_main:
                print(f"\n[{i}/{len(adapter_paths)}] {cp_path.name}")

            model = apply_checkpoint_adapter(base_model, str(cp_path), is_main)
            model.eval()

            evaluate_model(
                model, tokenizer, accelerator, cfg,
                all_prompts, all_answers, all_long_answers,
                adapter_label=str(cp_path),
            )

            del model
            torch.cuda.empty_cache()

    if is_main:
        print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()