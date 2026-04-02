# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Official implementation of "From Data to Rewards: a Bi-level Optimization Perspective on Maximum Likelihood Estimation" (arXiv:2510.07624). The active focus is on **LLM training via GRPO/SFT with learned embedding-based rewards**, applied to Countdown and PubMedQA tasks.

## Setup & Installation

```bash
pip install -e .                # Base package
pip install -e .[dev]           # Dev tools (ruff, black, pre-commit)
pip install -e .[llm]           # LLM training (transformers, trl, deepspeed, peft, lm_eval, etc.)
```

Python 3.12+ required. Build system: setuptools + setuptools_scm (version from git tags).

## Linting & Formatting

```bash
pre-commit run --all-files
```

Active hooks: **Ruff** (format + lint, line-length 88) and **Black**. No test suite — validation is done via eval scripts.

## LLM Training

Training scripts use **tyro** for CLI argument parsing from a `TrainConfig` dataclass. Run any script with `--help` to see all options.

### Countdown task (GRPO + SFT)

```bash
# GRPO — single GPU
python src/nll_to_po/llm/grpo_llm.py --model-name Qwen/Qwen3-1.7B --n-steps 400

# GRPO — multi-GPU with DeepSpeed ZeRO-3
accelerate launch --config_file config/deepspeed_zero3.yaml \
  src/nll_to_po/llm/grpo_llm.py --model-name Qwen/Qwen3-1.7B --n-steps 400

# SFT — single GPU
python src/nll_to_po/llm/sft_llm.py --model-name Qwen/Qwen3-4B --num-train-epochs 10
```

- Dataset: `HuggingFaceTB/Countdown-Task-GOLD`
- Models: Qwen3 family (0.6B, 1.7B, 4B, 8B)
- Reward functions: `format_reward_func`, `equation_reward_func`, `embedding_reward_func_constructor`
- LoRA targets: `q_proj`, `v_proj` (GRPO) / `k_proj`, `q_proj`, `v_proj` (SFT)

### Medical QA — PubMedQA (GRPO + SFT)

```bash
# Step 1: SFT
python src/nll_to_po/medical/sft_llm.py --model-name Qwen/Qwen3-0.6B --max-steps 400

# Step 2: GRPO with SFT adapter merged in
python src/nll_to_po/medical/grpo_llm.py \
  --model-name Qwen/Qwen3-0.6B \
  --adapter-path logs/Qwen3-0.6B/pubmed_qa/[peft][v15]trl-sft-TIMESTAMP \
  --n-steps 400

# Multi-GPU
accelerate launch --config_file config/deepspeed_zero3.yaml \
  src/nll_to_po/medical/grpo_llm.py --model-name Qwen/Qwen3-0.6B --n-steps 400
```

- Dataset: `bigbio/pubmed_qa`
- Embedding model: `NeuML/pubmedbert-base-embeddings-8M`
- Reward functions: `format_compliance_reward`, `answer_correctness_reward`, `embedding_reward_func_constructor_pubmedqa`
- Expected output format: `<answer>yes/no</answer><long_answer>...</long_answer>`
- Utilities in `medical/utils.py`: prompt formatting, answer extraction, reward functions

### Embedding covariance (U_star computation)

```bash
python -m nll_to_po.llm.embed_cov \
  --model_name google/embeddinggemma-300m \
  --dataset_name HuggingFaceTB/Countdown-Task-GOLD \
  --output_path results/cov/ \
  --is_sentence_transformer True
```

Computes `U* = (λ/2) * Σ^{-1}` (covariance) or `U* = (λn / 2Tr(Σ)) * I` (trace). Output: `trace.json` + `covariance.pt`.

## Evaluation Scripts

Eval scripts use **tyro** or **argparse** for CLI args.

### lm-eval-harness (`scripts/eval_harness.py`)

```bash
# Single adapter
python scripts/eval_harness.py --model-name Qwen/Qwen3-0.6B \
  --adapter-path logs/.../checkpoint-200 --tasks mmlu,medmcqa --gpu-ids 0

# Sweep checkpoints in parallel across GPUs
python scripts/eval_harness.py --model-name Qwen/Qwen3-0.6B \
  --adapter-base-path logs/.../run-dir --checkpoint-step 100 \
  --tasks mmlu,medmcqa --gpu-ids 0,1,2,3 --parallel-checkpoints 2
```

Key options: `--adapter-path` (single) vs `--adapter-base-path` (sweep over `checkpoint-*` dirs), `--start-checkpoint`, `--end-checkpoint`, `--checkpoint-step`, `--use-accelerate`, `--parallelize`.

### Countdown eval (`scripts/eval_countdown.py`)

```bash
python scripts/eval_countdown.py --model_name Qwen/Qwen3-0.6B \
  --adapter_path logs/.../checkpoint-200 --verbose 1
```

### PubMedQA eval (`scripts/eval_pubmed_accelerate.py`)

```bash
accelerate launch --num_processes 3 --multi_gpu scripts/eval_pubmed_accelerate.py \
  --model-name Qwen/Qwen3-0.6B --adapter-path logs/.../checkpoint-200 --batch-size 64
```

### Batch checkpoint eval (`scripts/eval_all_checkpoints.py`)

```bash
python scripts/eval_all_checkpoints.py --model_name Qwen/Qwen3-0.6B \
  --adapter_base_path logs/.../run-dir --gpu_ids 4,5,6 \
  --checkpoints "20,60,100,200,300,400" --batch_size 512
```

Batch eval scripts (`schedule.sh`, `schedule_8B.sh`) contain pre-configured runs for multiple seeds and model sizes.

## Architecture

### Active code (`src/nll_to_po/`)

- **`llm/grpo_llm.py`** — GRPO training on Countdown via TRL's `GRPOTrainer` with LoRA and multi-faceted rewards (format + equation correctness + embedding distance).
- **`llm/sft_llm.py`** — SFT baseline via TRL's `SFTTrainer`.
- **`llm/grpo_vlm.py`** — GRPO for Vision-Language Models (Qwen3-VL, Ministral) with 4-bit quantization.
- **`llm/embed_cov.py`** — `EmbeddingExtractor` and `SentenceTransformerEmbeddingExtractor` for computing covariance matrices and U_star. Supports `mean`/`last`/`cls` pooling.
- **`medical/grpo_llm.py`** — GRPO on PubMedQA with domain-specific embeddings and SFT adapter loading/merging.
- **`medical/sft_llm.py`** — SFT on PubMedQA with `completion_only_loss=True`.
- **`medical/utils.py`** — Prompt formatting (`format_prompt`), answer extraction (`extract_answer`, `extract_long_answer`), reward functions, and dataset preparation for both SFT and GRPO.
- **`training/reward.py`** — Reward abstractions used by LLM pipeline: `SentenceTransformerMahalanobisReward`, `AutoModelEmbeddingMahalanobisReward`, `BertEmbeddingMahalanobisReward`.

### Output directory convention

```
logs/{model_size}/{dataset}/[{embedder}][{u_star}][{pooling}][{lambda}][peft_{rank}][s:{steps}][{version}]trl-{method}-{timestamp}/
  ├── checkpoint-{step}/    # PEFT adapter weights
  └── runs/                 # TensorBoard logs
```

### Legacy code (no longer maintained)

The following modules were used for earlier non-LLM experiments (synthetic, tabular classification, MBRL) and are **not actively maintained**:

- `training/loss.py`, `training/data.py`, `training/utils.py` — Generic training loop, loss functions, data loading (UCI/Minari)
- `models/dn_policy.py` — MLP policy networks for regression/classification
- `bilevel/torchopt.py` — TorchOpt-based bilevel solver
- `scripts/mbrl.py`, `scripts/nll_mse.py`, `scripts/po_mse.py`, `scripts/po_rbf.py`, `scripts/llm.py` — Older experiment scripts
- `notebook/` — Paper figure notebooks (Sections 4.2, 5.2, 6.1, 6.2)
- Optional install extras: `.[mbrl]`, `.[classif]`, `.[bilevel]`
