#!/usr/bin/env python3
"""
Evaluate multiple LoRA checkpoints on the Countdown task.

Usage:
    python scripts/eval_all_checkpoints.py \
        --model_name Qwen/Qwen3-0.6B \
        --adapter_base_path "logs/Qwen3-0.6B/Countdown-Task-GOLD/[embeddinggemma-300m][Id][cls][0.001][peft_8][s:400][v13]trl-grpo-20260127-105022" \
        --gpu_ids 4,5,6 \
        --batch_size 256 \
        --checkpoint_step 100 \
        --start_checkpoint 100 \
        --end_checkpoint 1000 \
        --verbose 1
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import List, Optional
import re


def find_checkpoints(
    adapter_base_path: str,
    start_checkpoint: Optional[int] = None,
    end_checkpoint: Optional[int] = None,
    checkpoint_step: Optional[int] = None,
    specific_checkpoints: Optional[List[int]] = None,
) -> List[Path]:
    """
    Find all checkpoint directories in the adapter base path.

    Args:
        adapter_base_path: Base path containing checkpoint directories
        start_checkpoint: Minimum checkpoint number to evaluate
        end_checkpoint: Maximum checkpoint number to evaluate
        checkpoint_step: Step size between checkpoints (e.g., 100 for 100, 200, 300...)
        specific_checkpoints: List of specific checkpoint numbers to evaluate

    Returns:
        List of checkpoint paths sorted by checkpoint number
    """
    base_path = Path(adapter_base_path)

    if not base_path.exists():
        raise ValueError(f"Adapter base path does not exist: {adapter_base_path}")

    # Find all checkpoint directories
    checkpoint_dirs = []
    checkpoint_pattern = re.compile(r"checkpoint-(\d+)")

    for item in base_path.iterdir():
        if item.is_dir():
            match = checkpoint_pattern.match(item.name)
            if match:
                checkpoint_num = int(match.group(1))

                # Filter based on criteria
                if specific_checkpoints is not None:
                    if checkpoint_num in specific_checkpoints:
                        checkpoint_dirs.append((checkpoint_num, item))
                else:
                    include = True
                    if (
                        start_checkpoint is not None
                        and checkpoint_num < start_checkpoint
                    ):
                        include = False
                    if end_checkpoint is not None and checkpoint_num > end_checkpoint:
                        include = False
                    if (
                        checkpoint_step is not None
                        and checkpoint_num % checkpoint_step != 0
                    ):
                        include = False

                    if include:
                        checkpoint_dirs.append((checkpoint_num, item))

    # Sort by checkpoint number
    checkpoint_dirs.sort(key=lambda x: x[0])

    return [path for _, path in checkpoint_dirs]


def run_evaluation(
    model_name: str,
    adapter_path: str,
    gpu_ids: str,
    batch_size: int,
    verbose: int,
    results_file: str,
    num_processes: Optional[int] = None,
) -> dict:
    """
    Run evaluation for a single checkpoint.

    Returns:
        Dictionary with evaluation results
    """
    checkpoint_name = Path(adapter_path).name
    # results_file = Path(results_dir) / f"results_{checkpoint_name}.json"

    # # Create results directory if it doesn't exist
    # results_file.parent.mkdir(parents=True, exist_ok=True)

    # # Skip if results already exist
    # if results_file.exists():
    #     print(f"⏭️  Skipping {checkpoint_name} (results already exist)")
    #     with open(results_file, 'r') as f:
    #         return json.load(f)

    # Determine number of processes from GPU IDs if not specified
    if num_processes is None:
        num_processes = len(gpu_ids.split(","))

    # Build command
    cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        str(num_processes),
        "--multi_gpu",
        "scripts/eval_countdown_accelerate.py",
        "--model_name",
        model_name,
        "--adapter_path",
        adapter_path,
        "--verbose",
        str(verbose),
        "--save_results",
        str(results_file),
        "--batch_size",
        str(batch_size),
    ]

    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    print(f"\n{'=' * 80}")
    print(f"🚀 Evaluating: {checkpoint_name}")
    print(f"{'=' * 80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"GPUs: {gpu_ids}\n")

    # Run evaluation
    try:
        subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        print(f"✅ Completed: {checkpoint_name}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {checkpoint_name}")
        print(f"Error: {e.stderr}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple LoRA checkpoints on the Countdown task"
    )

    # Required arguments
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Base model name (e.g., Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--adapter_base_path",
        type=str,
        required=True,
        help="Base path containing checkpoint directories",
    )

    # GPU configuration
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help="Comma-separated GPU IDs (e.g., '4,5,6')",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of processes (defaults to number of GPUs)",
    )

    # Checkpoint selection
    parser.add_argument(
        "--start_checkpoint",
        type=int,
        default=None,
        help="Minimum checkpoint number to evaluate",
    )
    parser.add_argument(
        "--end_checkpoint",
        type=int,
        default=None,
        help="Maximum checkpoint number to evaluate",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=None,
        help="Step size between checkpoints (e.g., 100 for every 100 steps)",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma-separated list of specific checkpoint numbers (e.g., '100,200,500')",
    )

    # Evaluation configuration
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for evaluation"
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")

    # Output configuration
    parser.add_argument(
        "--results_file",
        type=str,
        default="results/results_countdown.json",
        help="File to save individual checkpoint results",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="results/all_checkpoints_summary.json",
        help="Path to save aggregated results summary",
    )

    args = parser.parse_args()

    # Parse specific checkpoints if provided
    specific_checkpoints = None
    if args.checkpoints:
        specific_checkpoints = [int(x.strip()) for x in args.checkpoints.split(",")]

    # Find checkpoints
    print("🔍 Searching for checkpoints...")
    checkpoints = find_checkpoints(
        args.adapter_base_path,
        start_checkpoint=args.start_checkpoint,
        end_checkpoint=args.end_checkpoint,
        checkpoint_step=args.checkpoint_step,
        specific_checkpoints=specific_checkpoints,
    )

    if not checkpoints:
        print("❌ No checkpoints found matching the criteria")
        return

    print(f"✅ Found {len(checkpoints)} checkpoints to evaluate:")
    for cp in checkpoints:
        print(f"   - {cp.name}")

    # Run evaluations
    results = []
    for i, checkpoint_path in enumerate(checkpoints, 1):
        print(f"\n[{i}/{len(checkpoints)}]")
        result = run_evaluation(
            model_name=args.model_name,
            adapter_path=str(checkpoint_path),
            gpu_ids=args.gpu_ids,
            batch_size=args.batch_size,
            verbose=args.verbose,
            results_file=args.results_file,
            num_processes=args.num_processes,
        )
        results.append(result)

    # Aggregate results
    print("\n" + "=" * 80)
    print("🎉 All evaluations complete!")


if __name__ == "__main__":
    main()
