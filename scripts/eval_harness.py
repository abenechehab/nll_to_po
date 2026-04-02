"""
Evaluate LLM checkpoints on standard benchmarks via lm-eval-harness.

Supports single-adapter evaluation, base-model evaluation, and sweeping
over multiple checkpoints in a training run directory.

Parallelism is handled at two levels:
  1. lm-eval-harness level: via ``--parallelize`` (model-parallel across
     visible GPUs) or by launching with ``accelerate`` (data-parallel).
  2. Script level: via ``--parallel-checkpoints`` which evaluates multiple
     checkpoints concurrently on disjoint GPU sets using multiprocessing.

Usage examples
--------------

# Single checkpoint, single GPU:
python eval_harness.py --model-name Qwen/Qwen3-0.6B \
    --adapter-path logs/.../checkpoint-200 \
    --tasks mmlu,medmcqa --gpu-ids 0

# Single checkpoint, data-parallel across 3 GPUs (accelerate):
python eval_harness.py --model-name Qwen/Qwen3-0.6B \
    --adapter-path logs/.../checkpoint-200 \
    --tasks mmlu,medmcqa --gpu-ids 0,1,2 --use-accelerate

# Sweep all checkpoints, 1 GPU each, sequentially:
python eval_harness.py --model-name Qwen/Qwen3-0.6B \
    --adapter-base-path logs/.../run-dir \
    --checkpoint-step 20 --tasks mmlu,medmcqa --gpu-ids 0

# Sweep checkpoints in parallel on 2 GPU groups (0,1) and (2,3):
python eval_harness.py --model-name Qwen/Qwen3-0.6B \
    --adapter-base-path logs/.../run-dir \
    --checkpoint-step 100 --tasks mmlu,medmcqa \
    --gpu-ids 0,1,2,3 --parallel-checkpoints 2
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import tyro


# ═══════════════════════════════════════════════════════════════════════════════
# CLI configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class HarnessEvalConfig:
    """Evaluate LLM checkpoints on lm-eval-harness benchmarks."""

    # ── model ────────────────────────────────────────────────────────────
    model_name: str
    """HuggingFace model id or local path to the base (pre-trained) model."""

    adapter_path: Optional[str] = None
    """Path to a single PEFT/LoRA adapter. Mutually exclusive with
    ``adapter_base_path``. If neither is set, the base model is evaluated."""

    # ── multi-checkpoint sweep ───────────────────────────────────────────
    adapter_base_path: Optional[str] = None
    """Root directory of a training run containing ``checkpoint-*`` sub-dirs."""

    start_checkpoint: Optional[int] = None
    """Minimum checkpoint step to include (inclusive)."""

    end_checkpoint: Optional[int] = None
    """Maximum checkpoint step to include (inclusive)."""

    checkpoint_step: Optional[int] = None
    """Only include checkpoints whose step is divisible by this value."""

    checkpoints: Optional[str] = None
    """Comma-separated list of specific checkpoint step numbers
    (e.g. '100,200,500'). Overrides start/end/step filters."""

    # ── lm-eval-harness settings ─────────────────────────────────────────
    tasks: str = "mmlu,medmcqa"
    """Comma-separated task names passed to ``lm_eval --tasks``."""

    num_fewshot: Optional[int] = None
    """Number of few-shot examples. If None, uses the task default."""

    batch_size: str = "auto"
    """Batch size for lm-eval-harness. ``'auto'`` lets it pick optimally."""

    max_length: Optional[int] = None
    """Override model max sequence length for evaluation."""

    apply_chat_template: bool = False
    """Pass ``--apply_chat_template`` to lm-eval (for instruct models)."""

    log_samples: bool = True
    """Pass ``--log_samples`` to save per-sample predictions."""

    extra_model_args: Optional[str] = None
    """Additional comma-separated key=value model_args
    (e.g. 'trust_remote_code=True,load_in_4bit=True')."""

    # ── GPU / parallelism ────────────────────────────────────────────────
    gpu_ids: str = "0"
    """Comma-separated GPU IDs to use (e.g. '0,1,2').
    For single-checkpoint eval, all listed GPUs serve one evaluation.
    For parallel checkpoint sweeps, GPUs are split across workers."""

    use_accelerate: bool = False
    """Launch lm_eval via ``accelerate launch`` for data-parallel evaluation.
    Each GPU gets a full copy of the model and processes a shard of the data.
    Best for small models that fit on a single GPU."""

    parallelize: bool = False
    """Pass ``parallelize=True`` in model_args for model-parallel evaluation.
    The model is sharded across all visible GPUs.
    Best for large models that do NOT fit on a single GPU."""

    parallel_checkpoints: int = 1
    """Number of checkpoints to evaluate concurrently (multiprocessing).
    GPUs are split evenly across workers. E.g. with ``--gpu-ids 0,1,2,3``
    and ``--parallel-checkpoints 2``, worker 0 gets GPUs 0,1 and worker 1
    gets GPUs 2,3."""

    # ── output ───────────────────────────────────────────────────────────
    output_path: str = "lm_eval_results"
    """Root directory where lm-eval-harness writes its result files."""

    save_results: Optional[str] = None
    """Path to a JSON file where aggregated results are appended (one entry
    per checkpoint). If None, results are only stored in ``output_path``."""

    verbose: int = 1
    """Verbosity level. 0 = minimal, 1 = show scores, 2 = show full stdout."""


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint discovery (shared with eval_pubmed_accelerate.py)
# ═══════════════════════════════════════════════════════════════════════════════

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def find_checkpoints(cfg: HarnessEvalConfig) -> list[Path]:
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
# Build lm_eval model_args string
# ═══════════════════════════════════════════════════════════════════════════════


def _build_model_args(
    cfg: HarnessEvalConfig,
    adapter_path: str | None,
) -> str:
    """Construct the ``--model_args`` string for lm_eval CLI."""
    parts: list[str] = [
        f"pretrained={cfg.model_name}",
        "dtype=bfloat16",
    ]

    # PEFT adapter
    if adapter_path is not None:
        # Check if there's an adapter_config.json (confirms it's a PEFT dir)
        adapter_config = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(adapter_config):
            parts.append(f"peft={adapter_path}")
        else:
            # Treat as a full model checkpoint (merged weights)
            parts[0] = f"pretrained={adapter_path}"

    # Model parallelism
    if cfg.parallelize:
        parts.append("parallelize=True")

    # Max length override
    if cfg.max_length is not None:
        parts.append(f"max_length={cfg.max_length}")

    # Extra user-provided args
    if cfg.extra_model_args:
        parts.append(cfg.extra_model_args)

    return ",".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Parse lm-eval-harness result files
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_results(result_dir: str, verbose: int = 0) -> dict[str, float]:
    """Find the most recent results_*.json in ``result_dir`` and extract scores."""
    # Use Path.rglob instead of glob.glob — the latter interprets [ ] as
    # character-class wildcards, which silently breaks on directory names
    # like "[peft][v15]trl-sft-...".
    result_files = list(Path(result_dir).rglob("results_*.json"))

    if not result_files:
        if verbose >= 1:
            print(f"  [debug] No results_*.json found under: {result_dir}")
            # List what IS there to help diagnose
            try:
                contents = list(Path(result_dir).rglob("*"))
                print(f"  [debug] Directory contents ({len(contents)} items):")
                for p in contents[:20]:
                    print(f"           {p}")
            except Exception:
                pass
        return {}

    latest_file = max(result_files, key=os.path.getmtime)
    with open(latest_file) as f:
        data = json.load(f)

    scores: dict[str, float] = {}
    for task_name, task_results in data.get("results", {}).items():
        # Prefer acc_norm, fall back to acc
        acc = task_results.get("acc_norm,none", task_results.get("acc,none", None))
        if acc is not None:
            scores[task_name] = acc
        # Also collect other useful metrics
        for metric_key in ("exact_match,none", "f1,none", "mc2,none"):
            val = task_results.get(metric_key)
            if val is not None:
                scores[f"{task_name}/{metric_key.split(',')[0]}"] = val

    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# Run a single lm-eval evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def run_single_eval(
    cfg: HarnessEvalConfig,
    adapter_path: str | None,
    gpu_ids: str,
) -> dict:
    """Run lm-eval-harness for one model/adapter and return parsed scores.

    This function is designed to be safe for use in a subprocess worker.
    """
    label = Path(adapter_path).name if adapter_path else "base_model"

    # ── output directory ─────────────────────────────────────────────────
    if adapter_path is not None:
        # Mirror the adapter directory structure under output_path
        result_dir = os.path.join(cfg.output_path, *Path(adapter_path).parts[1:])
    else:
        result_dir = os.path.join(
            cfg.output_path, cfg.model_name.replace("/", "_"), "base_model"
        )
    os.makedirs(result_dir, exist_ok=True)

    # ── build model_args ─────────────────────────────────────────────────
    model_args = _build_model_args(cfg, adapter_path)

    # ── build command ────────────────────────────────────────────────────
    cmd: list[str] = []

    if cfg.use_accelerate:
        num_procs = len(gpu_ids.split(","))
        cmd += [
            "accelerate",
            "launch",
            "--num_processes",
            str(num_procs),
            "--multi_gpu",
            "-m",
            "lm_eval",
        ]
    else:
        cmd += ["lm_eval"]

    cmd += [
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        cfg.tasks,
        "--batch_size",
        cfg.batch_size,
        "--output_path",
        result_dir,
    ]

    if cfg.num_fewshot is not None:
        cmd += ["--num_fewshot", str(cfg.num_fewshot)]

    if cfg.apply_chat_template:
        cmd += ["--apply_chat_template"]

    if cfg.log_samples:
        cmd += ["--log_samples"]

    # ── environment ──────────────────────────────────────────────────────
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    # Silence tokenizer parallelism warnings in subprocess
    env["TOKENIZERS_PARALLELISM"] = "false"

    # ── run ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  Evaluating: {label}")
    print(f"  GPUs: {gpu_ids}")
    print(f"  Tasks: {cfg.tasks}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=False,  # We handle errors ourselves
        )

        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  FAILED: {label} (exit code {result.returncode})")
            stderr_tail = result.stderr[-1000:] if result.stderr else ""
            print(f"  stderr: {stderr_tail}")
            return {
                "adapter_path": adapter_path,
                "label": label,
                "error": f"exit code {result.returncode}",
                "stderr": stderr_tail,
            }

        # Show stdout tail
        if cfg.verbose >= 2:
            print(result.stdout)
        elif cfg.verbose >= 1:
            # Show just the last section with the results table
            stdout_lines = result.stdout.strip().split("\n")
            # lm-eval prints a table at the end; grab last ~30 lines
            tail = "\n".join(stdout_lines[-30:])
            print(tail)

    except FileNotFoundError:
        return {
            "adapter_path": adapter_path,
            "label": label,
            "error": "lm_eval command not found. Install with: pip install lm-eval",
        }

    # ── parse results ────────────────────────────────────────────────────
    scores = _parse_results(result_dir, verbose=cfg.verbose)

    record = {
        "config": {
            "model_name": cfg.model_name,
            "adapter_path": adapter_path,
            "tasks": cfg.tasks,
            "gpu_ids": gpu_ids,
            "batch_size": cfg.batch_size,
            "num_fewshot": cfg.num_fewshot,
            "use_accelerate": cfg.use_accelerate,
            "parallelize": cfg.parallelize,
        },
        "metrics": scores,
        "elapsed_seconds": round(elapsed, 1),
        "label": label,
        "result_dir": result_dir,
    }

    # ── print summary ────────────────────────────────────────────────────
    print(f"\n  {label} completed in {elapsed:.0f}s")
    if scores:
        max_key_len = max(len(k) for k in scores)
        for task_name, acc in sorted(scores.items()):
            print(f"    {task_name:<{max_key_len}}  {acc:.4f}")
    else:
        print("    (no scores parsed — check result_dir for raw outputs)")

    # ── append to aggregated results file ────────────────────────────────
    if cfg.save_results:
        _append_result(cfg.save_results, record)

    return record


def _append_result(path: str, record: dict) -> None:
    """Thread/process-safe append of a result record to a JSON list file."""
    results_path = Path(path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Simple file-level locking via a .lock file
    lock_path = results_path.with_suffix(".lock")
    import fcntl

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            if results_path.exists():
                with open(results_path, "r") as f:
                    data = json.load(f)
            else:
                data = []

            data.append(record)

            with open(results_path, "w") as f:
                json.dump(data, f, indent=2)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


# ═══════════════════════════════════════════════════════════════════════════════
# GPU allocation helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _split_gpus(gpu_ids_str: str, n_workers: int) -> list[str]:
    """Split a comma-separated GPU list into ``n_workers`` groups.

    Returns a list of comma-separated GPU ID strings, one per worker.
    E.g. _split_gpus("0,1,2,3", 2) -> ["0,1", "2,3"]
    """
    ids = [g.strip() for g in gpu_ids_str.split(",")]
    if n_workers > len(ids):
        raise ValueError(
            f"parallel_checkpoints={n_workers} exceeds available GPUs ({len(ids)}). "
            f"Need at least one GPU per worker."
        )

    # Distribute as evenly as possible
    base_size = len(ids) // n_workers
    remainder = len(ids) % n_workers
    groups: list[str] = []
    idx = 0
    for w in range(n_workers):
        size = base_size + (1 if w < remainder else 0)
        groups.append(",".join(ids[idx : idx + size]))
        idx += size
    return groups


# ═══════════════════════════════════════════════════════════════════════════════
# Worker function for parallel checkpoint evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def _worker_eval(
    cfg: HarnessEvalConfig,
    adapter_path: str | None,
    gpu_ids: str,
) -> dict:
    """Thin wrapper around ``run_single_eval`` for use with ProcessPoolExecutor."""
    return run_single_eval(cfg, adapter_path, gpu_ids)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    cfg = tyro.cli(HarnessEvalConfig)

    # ── determine which adapters to evaluate ─────────────────────────────
    eval_targets: list[str | None]  # None means base model

    if cfg.adapter_base_path is not None:
        checkpoints = find_checkpoints(cfg)
        if not checkpoints:
            print("No checkpoints found matching the given criteria.")
            return
        print(f"Found {len(checkpoints)} checkpoint(s) to evaluate:")
        for cp in checkpoints:
            print(f"  - {cp.name}")
        eval_targets = [str(cp) for cp in checkpoints]

    elif cfg.adapter_path is not None:
        eval_targets = [cfg.adapter_path]

    else:
        eval_targets = [None]  # base model

    # ── sequential or parallel execution ─────────────────────────────────
    all_results: list[dict] = []

    if cfg.parallel_checkpoints > 1 and len(eval_targets) > 1:
        # Parallel checkpoint evaluation across disjoint GPU sets
        n_workers = min(cfg.parallel_checkpoints, len(eval_targets))
        gpu_groups = _split_gpus(cfg.gpu_ids, n_workers)

        print(f"\nParallel evaluation: {n_workers} workers")
        for i, gpus in enumerate(gpu_groups):
            print(f"  Worker {i}: GPUs [{gpus}]")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Round-robin assignment: submit each checkpoint to the next
            # available worker's GPU group.
            futures = {}
            for idx, target in enumerate(eval_targets):
                worker_gpus = gpu_groups[idx % n_workers]
                future = executor.submit(_worker_eval, cfg, target, worker_gpus)
                futures[future] = target

            for future in as_completed(futures):
                target = futures[future]
                label = Path(target).name if target else "base_model"
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    print(f"  Worker exception for {label}: {e}")
                    all_results.append({"label": label, "error": str(e)})

    else:
        # Sequential evaluation — all GPUs serve one checkpoint at a time
        for i, target in enumerate(eval_targets, 1):
            label = Path(target).name if target else "base_model"
            print(f"\n[{i}/{len(eval_targets)}] {label}")
            result = run_single_eval(cfg, target, cfg.gpu_ids)
            all_results.append(result)

    # ── final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    # Collect all task names across results for a consistent table
    all_task_names: set[str] = set()
    for r in all_results:
        if "metrics" in r:
            all_task_names.update(r["metrics"].keys())
    task_names_sorted = sorted(all_task_names)

    if task_names_sorted:
        # Header
        label_width = max(len(r.get("label", "?")) for r in all_results)
        label_width = max(label_width, 10)
        header = f"{'Checkpoint':<{label_width}}"
        for tn in task_names_sorted:
            header += f"  {tn:>12}"
        header += f"  {'Time':>7}"
        print(header)
        print("-" * len(header))

        for r in all_results:
            label = r.get("label", "?")
            if "error" in r:
                print(f"{label:<{label_width}}  ERROR: {r['error'][:60]}")
                continue
            row = f"{label:<{label_width}}"
            for tn in task_names_sorted:
                val = r.get("metrics", {}).get(tn)
                row += f"  {val:>12.4f}" if val is not None else f"  {'—':>12}"
            elapsed = r.get("elapsed_seconds", 0)
            row += f"  {elapsed:>6.0f}s"
            print(row)
    else:
        for r in all_results:
            label = r.get("label", "?")
            if "error" in r:
                print(f"  {label}: ERROR — {r['error'][:80]}")
            else:
                print(f"  {label}: (no scores parsed)")

    print("=" * 80)
    print("All evaluations complete.")

    if cfg.save_results:
        print(f"Aggregated results saved to: {cfg.save_results}")


if __name__ == "__main__":
    main()
