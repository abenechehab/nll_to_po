"""
Distributed evaluation script optimized for large memory GPUs
Usage:
    accelerate launch --num_processes 3 --multi_gpu evaluate_countdown_distributed.py \
        --model_name Qwen/Qwen2.5-0.6B \
        --adapter_path ./path/to/adapter \
        --batch_size 128
"""

import json
import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import numpy as np
from accelerate import Accelerator

from nll_to_po.training.reward import equation_reward_func


def prepare_prompts(dataset, tokenizer):
    """Prepare all prompts in advance"""
    prompts = []
    targets = []
    numbers = []

    for example in dataset:
        prompt = [
            {
                "role": "system",
                "content": example["prompt"][0]["content"],
            },
            {
                "role": "user",
                "content": example["prompt"][1]["content"],
            },
            {
                "role": "assistant",
                "content": "Let me solve this step by step.\n<think>",
            },
        ]
        prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=False
        )

        prompts.append(prompt)
        targets.append(example["target"])
        numbers.append(example["nums"])

    return prompts, targets, numbers


def generate_batch(model, tokenizer, prompts, max_new_tokens=512):
    """Generate completions for a batch"""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
        return_attention_mask=True,
    ).to(model.device)

    with torch.no_grad():
        # Access underlying model if wrapped in DDP
        model_to_use = model.module if hasattr(model, "module") else model

        outputs = model_to_use.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for evaluation
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,  # Important for speed
        )

    completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Remove prompts (handle tokenizer chat template properly)
    completions_clean = []
    for completion, prompt in zip(completions, prompts):
        if completion.startswith(prompt):
            completions_clean.append(completion[len(prompt) :])
        else:
            # Fallback: find where assistant response starts
            completions_clean.append(
                completion.split("<think>")[-1]
                if "<think>" in completion
                else completion
            )

    return completions_clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--adapter_path", type=str, default=None, help="Path to PEFT adapter (optional)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size per GPU"
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--save_results", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    args = parser.parse_args()

    # Initialize accelerator
    accelerator = Accelerator()

    # Load model and tokenizer
    if accelerator.is_main_process:
        print(f"Loading model: {args.model_name}")
        if args.adapter_path:
            print(f"Will load adapter from: {args.adapter_path}")
        else:
            print("Evaluating base model (no adapter)")
        print(f"Using {accelerator.num_processes} GPUs")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * accelerator.num_processes}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set padding side to left for generation
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.device},  # Put model on assigned device
    )

    # Load adapter if provided
    if args.adapter_path:
        if accelerator.is_main_process:
            print("Loading PEFT adapter...")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        model = model.merge_and_unload()
    else:
        model = base_model

    model.eval()

    # Load dataset
    if accelerator.is_main_process:
        print("Loading Countdown-Task-GOLD dataset...")

    dataset = load_dataset("HuggingFaceTB/Countdown-Task-GOLD", "test")["test"]

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    if accelerator.is_main_process:
        print(f"Dataset size: {len(dataset)}")

    # Prepare all prompts
    if accelerator.is_main_process:
        print("Preparing prompts...")

    all_prompts, all_targets, all_numbers = prepare_prompts(dataset, tokenizer)

    # Split data across processes
    with accelerator.split_between_processes(
        list(zip(all_prompts, all_targets, all_numbers))
    ) as split_data:
        process_prompts = [item[0] for item in split_data]
        process_targets = [item[1] for item in split_data]
        process_numbers = [item[2] for item in split_data]

    # Generate completions
    if accelerator.is_main_process:
        print("\nGenerating completions...")
        print(f"This process will handle {len(process_prompts)} examples")

    process_completions = []

    for i in tqdm(
        range(0, len(process_prompts), args.batch_size),
        desc=f"GPU {accelerator.process_index}",
        disable=not accelerator.is_main_process,
    ):
        batch_prompts = process_prompts[i : i + args.batch_size]
        batch_completions = generate_batch(
            model, tokenizer, batch_prompts, max_new_tokens=args.max_new_tokens
        )
        process_completions.extend(batch_completions)

    # Gather all results
    all_completions = accelerator.gather_for_metrics(process_completions)
    all_targets = accelerator.gather_for_metrics(process_targets)
    all_numbers = accelerator.gather_for_metrics(process_numbers)

    # Evaluate on main process
    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("Evaluating completions...")

        rewards = equation_reward_func(
            all_completions, all_targets, all_numbers, verbose=args.verbose
        )

        accuracy = np.mean(rewards)
        num_correct = sum(rewards)
        total = len(rewards)

        has_answer_tag = sum(
            1 for c in all_completions if "<answer>" in c and "</answer>" in c
        )
        answer_tag_rate = has_answer_tag / total

        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"Model: {args.model_name}")
        print(
            f"Adapter: {args.adapter_path if args.adapter_path else 'None (base model)'}"
        )
        print(f"Total examples: {total}")
        print(f"Accuracy: {accuracy:.2%} ({num_correct}/{total})")
        print(f"Answer tag format rate: {answer_tag_rate:.2%}")
        print("=" * 80)

        # Save results
        if args.save_results:
            detailed_results = {
                "config": {
                    "model_name": args.model_name,
                    "adapter_path": args.adapter_path,
                    "num_gpus": accelerator.num_processes,
                    "batch_size_per_gpu": args.batch_size,
                    "effective_batch_size": args.batch_size * accelerator.num_processes,
                },
                "metrics": {
                    "accuracy": float(accuracy),
                    "num_correct": int(num_correct),
                    "total": total,
                    "answer_tag_rate": float(answer_tag_rate),
                },
                "examples": [
                    {
                        "completion": comp,
                        "target": tgt,
                        "numbers": nums,
                        "correct": bool(reward == 1.0),
                    }
                    for comp, tgt, nums, reward in zip(
                        all_completions[:5],  # Save first 5 examples
                        all_targets[:5],
                        all_numbers[:5],
                        rewards[:5],
                    )
                ],
            }

            if Path(args.save_results).exists():
                with open(args.save_results, "r") as f:
                    data = json.load(f)
            else:
                data = []

            data.append(detailed_results)

            with open(args.save_results, "w") as f:
                json.dump(data, f, indent=2)

            print(f"\nDetailed results saved to: {args.save_results}")

        # Show sample predictions
        if args.verbose > 0:
            print("\n" + "=" * 80)
            print("SAMPLE PREDICTIONS")
            print("=" * 80)
            for i in range(min(5, len(all_completions))):
                print(f"\n--- Example {i + 1} ---")
                print(f"Numbers: {all_numbers[i]}, Target: {all_targets[i]}")
                print(f"Completion: {all_completions[i][:5]}...")
                print(f"Correct: {rewards[i] == 1.0}")


if __name__ == "__main__":
    main()
