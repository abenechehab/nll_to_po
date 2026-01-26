"""
Evaluation script for fine-tuned model on Countdown-Task-GOLD dataset.

Usage:
    python evaluate_countdown.py --model_name Qwen/Qwen2.5-0.6B --adapter_path ./path/to/adapter
"""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import numpy as np

from nll_to_po.training.reward import equation_reward_func


def load_model_and_adapter(model_name, adapter_path):
    """Load base model and apply PEFT adapter"""
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading PEFT adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()  # Merge adapter weights for faster inference
    model.eval()

    return model, tokenizer


def generate_completion(model, tokenizer, prompt, max_new_tokens=512):
    """Generate a single completion"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=False,  # Greedy decoding for evaluation
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the completion
    completion = completion[len(prompt) :]

    return completion


def evaluate_model(
    model, tokenizer, test_dataset, batch_size=1, max_samples=None, verbose=0
):
    """
    Evaluate model on test dataset

    Args:
        model: The fine-tuned model
        tokenizer: Tokenizer
        test_dataset: Test split of the dataset
        batch_size: Currently only supports 1
        max_samples: Maximum number of samples to evaluate (None for all)
        verbose: Verbosity level for equation_reward_func

    Returns:
        dict: Evaluation metrics
    """
    if max_samples is not None:
        test_dataset = test_dataset.select(range(min(max_samples, len(test_dataset))))

    all_completions = []
    all_targets = []
    all_numbers = []

    print(f"Evaluating on {len(test_dataset)} examples...")

    for example in tqdm(test_dataset, desc="Generating completions"):
        # Format prompt
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

        # Extract numbers and target
        numbers, target = example["nums"], example["target"]

        # Generate completion
        completion = generate_completion(model, tokenizer, prompt)

        all_completions.append(completion)
        all_targets.append(target)
        all_numbers.append(numbers)

        if verbose > 1:
            print(f"\n{'=' * 80}")
            print(f"Prompt: {prompt}")
            print(f"Completion: {completion}")
            print(f"Target: {target}")
            print(f"Numbers: {numbers}")

    # Evaluate all completions
    print("\nEvaluating completions...")
    rewards = equation_reward_func(
        all_completions, all_targets, all_numbers, verbose=verbose
    )

    # Compute metrics
    accuracy = np.mean(rewards)
    num_correct = sum(rewards)
    total = len(rewards)

    # Additional metrics
    has_answer_tag = sum(
        1 for c in all_completions if "<answer>" in c and "</answer>" in c
    )
    answer_tag_rate = has_answer_tag / total

    results = {
        "accuracy": accuracy,
        "num_correct": int(num_correct),
        "total": total,
        "answer_tag_rate": answer_tag_rate,
        "rewards": rewards,
    }

    return results, all_completions, all_targets, all_numbers


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model on Countdown-Task-GOLD"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Base model name (e.g., Qwen/Qwen2.5-0.6B)",
    )
    parser.add_argument(
        "--adapter_path", type=str, required=True, help="Path to PEFT adapter"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Maximum samples to evaluate"
    )
    parser.add_argument(
        "--verbose", type=int, default=0, help="Verbosity level (0, 1, 2)"
    )
    parser.add_argument(
        "--save_results",
        type=str,
        default=None,
        help="Path to save detailed results (optional)",
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_and_adapter(args.model_name, args.adapter_path)

    # Load dataset
    print("Loading Countdown-Task-GOLD dataset...")
    dataset = load_dataset("HuggingFaceTB/Countdown-Task-GOLD", "test")["test"]

    # Inspect dataset structure
    print("\nDataset info:")
    print(f"  Total examples: {len(dataset)}")
    print(f"  Column names: {dataset.column_names}")
    print("\nFirst example:")
    print(dataset[0])

    # Evaluate
    results, completions, targets, numbers = evaluate_model(
        model, tokenizer, dataset, max_samples=args.max_samples, verbose=args.verbose
    )

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(
        f"Accuracy: {results['accuracy']:.2%} ({results['num_correct']}/{results['total']})"
    )
    print(f"Answer tag format rate: {results['answer_tag_rate']:.2%}")
    print("=" * 80)

    # Save detailed results if requested
    if args.save_results:
        import json

        detailed_results = {
            "metrics": {
                "accuracy": results["accuracy"],
                "num_correct": results["num_correct"],
                "total": results["total"],
                "answer_tag_rate": results["answer_tag_rate"],
            },
            "examples": [
                {
                    "completion": comp,
                    "target": tgt,
                    "numbers": nums,
                    "correct": reward == 1.0,
                }
                for comp, tgt, nums, reward in zip(
                    completions, targets, numbers, results["rewards"]
                )
            ],
        }

        with open(args.save_results, "w") as f:
            json.dump(detailed_results, f, indent=2)
        print(f"\nDetailed results saved to: {args.save_results}")

    # Show some examples
    if args.verbose > 0:
        print("\n" + "=" * 80)
        print("SAMPLE PREDICTIONS")
        print("=" * 80)
        for i in range(min(5, len(completions))):
            print(f"\nExample {i + 1}:")
            print(f"Numbers: {numbers[i]}, Target: {targets[i]}")
            print(f"Completion: {completions[i]}")
            print(f"Correct: {results['rewards'][i] == 1.0}")


if __name__ == "__main__":
    main()
