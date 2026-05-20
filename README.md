<div align="center">
<h1>Bridging SFT and RL: Learning implicit rewards from supervised data</h1>

<!-- [![paper](https://img.shields.io/static/v1?label=arXiv&message=2402.03885&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2510.07624) --> 
<!-- [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/papers/2510.07624) --> 
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/license/MIT)
[![Python: 3.10](https://img.shields.io/badge/Python-3.12-blue)]()

</div>

This repository contains the official implementation of the paper:

   >Abdelhakim Benechehab, Gabriel Singer, Youssef Attia El Hili, Corentin Léger, Giuseppe Paolo, Albert Thomas, Maurizio Filippone, Balázs Kégl.
   [Bridging SFT and RL: Learning implicit rewards from supervised data]().

### 📝 Abstract:
Reinforcement learning (RL) has emerged as a powerful paradigm for adapting large language models (LLM), offering advantages over supervised fine-tuning (SFT) including reduced catastrophic forgetting and improved generalization. However, these benefits require explicit reward signals, often obtained from human preferences or verifiable outcomes, which are unavailable in many cases. We address this gap by introducing a framework that derives reward functions directly from supervised data, enabling RL-based training without additional annotation. Our approach formulates reward learning as a bilevel optimization problem: the outer level optimizes the reward parameters to maximize likelihood on the training data, while the inner level trains a policy via entropy-regularized RL. Theoretical analysis in a tractable Gaussian setting reveals that the optimal reward takes the form of a negative Mahalanobis distance, scaled by the inverse covariance of the target distribution. We extend this insight to language models by defining rewards as distances in a pretrained embedding space. Our experiments on LLM fine-tuning demonstrate that our learned rewards match the performance of oracle RL that have access to ground-truth rewards. Additional results on synthetic data, tabular classification and dynamics modeling further validate the generality of our framework.

Which is an extension of the following [ICLR26 Workshop](https://openreview.net/group?id=ICLR.cc/2026/Workshop/LLM_Reasoning#tab-accept) paper:

   >Abdelhakim Benechehab, Youssef Attia El Hili, Albert Thomas, Giuseppe Paolo, Maurizio Filippone.
   [Embedding Distance as a Reward Signal can replace Verifiers for LLM Reasoning](https://openreview.net/attachment?id=5P1F5sSFjN&name=pdf).

## 🚀 Installation

🔹 Create and activate a Python 3.12 environment (conda, micromamba, or venv):
```bash
conda create -n nllpo python=3.12
conda activate nllpo
```

🔹 Install the base package (editable mode):
```bash
pip install -e .
```

🔹 **For LLM training and evaluation** (GRPO, SFT, eval scripts):
```bash
pip install -e .[llm]
```

🔹 **For developers** (pre-commit hooks for Ruff + Black):
```bash
pip install -e .[dev]
pre-commit install
```

## 🤖 LLM Applications

This repository implements GRPO and SFT training with learned embedding-based rewards, applied to two tasks: **Countdown** (mathematical reasoning) and **PubMedQA** (medical question answering). All training scripts are fully CLI-configurable via [tyro](https://brentyi.github.io/tyro/) — run any script with `--help` to see all options.

### Countdown (mathematical reasoning)

**GRPO training** — learns a reward via Mahalanobis distance in embedding space:

```bash
# Single GPU
python src/nll_to_po/llm/grpo_llm.py --help

python src/nll_to_po/llm/grpo_llm.py \
  --model-name Qwen/Qwen3-1.7B \
  --embed True \
  --reward-embedding-model google/embeddinggemma-300m \
  --lam 0.001 \
  --u-star-type id \
  --n-steps 400

# Multi-GPU with DeepSpeed ZeRO-3 (recommended for ≥4 GPUs)
accelerate launch --config_file config/deepspeed_zero3.yaml \
  src/nll_to_po/llm/grpo_llm.py \
  --model-name Qwen/Qwen3-1.7B \
  --embed True \
  --reward-embedding-model google/embeddinggemma-300m \
  --n-steps 400

# Multi-GPU without DeepSpeed (e.g. 2–3 GPUs)
accelerate launch --num_processes 2 --multi_gpu \
  src/nll_to_po/llm/grpo_llm.py \
  --model-name Qwen/Qwen3-1.7B \
  --n-steps 400
```

**SFT baseline:**

```bash
# Single GPU
python src/nll_to_po/llm/sft_llm.py \
  --model-name Qwen/Qwen3-1.7B \
  --num-train-epochs 10

# Multi-GPU with DeepSpeed ZeRO-3
accelerate launch --config_file config/deepspeed_zero3.yaml \
  src/nll_to_po/llm/sft_llm.py \
  --model-name Qwen/Qwen3-1.7B \
  --num-train-epochs 10
```

### PubMedQA (medical question answering)

The recommended pipeline runs SFT first, then GRPO with the SFT adapter merged in.

**Step 1 — SFT:**

```bash
# Single GPU
python src/nll_to_po/medical/sft_llm.py \
  --model-name Qwen/Qwen3-0.6B \
  --max-steps 400

# Multi-GPU with DeepSpeed ZeRO-3
accelerate launch --config_file config/deepspeed_zero3.yaml \
  src/nll_to_po/medical/sft_llm.py \
  --model-name Qwen/Qwen3-0.6B \
  --max-steps 400
```

**Step 2 — GRPO** (pass the SFT checkpoint via `--adapter-path`):

```bash
# Single GPU
python src/nll_to_po/medical/grpo_llm.py \
  --model-name Qwen/Qwen3-0.6B \
  --adapter-path logs/Qwen3-0.6B/pubmed_qa/[peft][v15]trl-sft-TIMESTAMP \
  --embed True \
  --reward-embedding-model NeuML/pubmedbert-base-embeddings-8M \
  --n-steps 400

# Multi-GPU with DeepSpeed ZeRO-3
accelerate launch --config_file config/deepspeed_zero3.yaml \
  src/nll_to_po/medical/grpo_llm.py \
  --model-name Qwen/Qwen3-0.6B \
  --adapter-path logs/Qwen3-0.6B/pubmed_qa/[peft][v15]trl-sft-TIMESTAMP \
  --embed True \
  --n-steps 400
```

### U_star precomputation (optional, improves reward scaling)

Before training with `--u-star-type cov` or `--u-star-type trace`, precompute the embedding covariance:

```bash
python -m nll_to_po.llm.embed_cov \
  --model_name google/embeddinggemma-300m \
  --dataset_name HuggingFaceTB/Countdown-Task-GOLD \
  --output_path results/cov/ \
  --is_sentence_transformer True
```

### Evaluation

```bash
# Countdown task accuracy (single adapter)
python scripts/eval_countdown.py \
  --model_name Qwen/Qwen3-1.7B \
  --adapter_path logs/.../checkpoint-400

# Countdown task — sweep all checkpoints in a run directory
python scripts/eval_all_checkpoints.py \
  --model_name Qwen/Qwen3-1.7B \
  --adapter_base_path logs/.../run-dir \
  --checkpoint_step 20 \
  --gpu_ids 0,1,2

# PubMedQA accuracy (multi-GPU)
accelerate launch --num_processes 3 --multi_gpu \
  scripts/eval_pubmed_accelerate.py \
  --model-name Qwen/Qwen3-0.6B \
  --adapter-path logs/.../checkpoint-400 \
  --batch-size 64

# Standard benchmarks via lm-eval-harness (e.g. MMLU, MedMCQA)
python scripts/eval_harness.py \
  --model-name Qwen/Qwen3-0.6B \
  --adapter-path logs/.../checkpoint-400 \
  --tasks mmlu,medmcqa \
  --gpu-ids 0
```

Checkpoints are saved to `logs/{model}/{dataset}/{tags}trl-{method}-{timestamp}/checkpoint-{step}/`. TensorBoard logs are in the same directory under `runs/`.

## 📁 Project Structure

```
├── src/nll_to_po/
│   ├── llm/
│   │   ├── grpo_llm.py         # GRPO training — Countdown
│   │   ├── sft_llm.py          # SFT training — Countdown
│   │   ├── grpo_vlm.py         # GRPO training — Vision-Language Models
│   │   └── embed_cov.py        # Embedding covariance / U_star computation
│   ├── medical/
│   │   ├── grpo_llm.py         # GRPO training — PubMedQA
│   │   ├── sft_llm.py          # SFT training — PubMedQA
│   │   └── utils.py            # Prompt formatting, reward functions, dataset prep
│   └── training/
│       └── reward.py           # Reward abstractions (Mahalanobis, embedding-based)
├── scripts/
│   ├── eval_countdown.py       # Countdown task evaluation
│   ├── eval_all_checkpoints.py # Batch checkpoint evaluation (Countdown)
│   ├── eval_pubmed_accelerate.py  # PubMedQA evaluation (multi-GPU)
│   └── eval_harness.py         # Standard benchmarks via lm-eval-harness
└── config/
    └── deepspeed_zero3.yaml    # Accelerate + DeepSpeed ZeRO-3 config (8 GPUs)
```

## 🧪 Paper Experiments

The following notebooks and scripts reproduce the results from the paper. They are no longer actively maintained.

### Section 4.2 — Synthetic data
* [`notebook/4-2-fig1-synthetic.ipynb`](notebook/4-2-fig1-synthetic.ipynb)
* [`notebook/4-2-fig2-distribution.ipynb`](notebook/4-2-fig2-distribution.ipynb)

### Section 5.2 — Implicit differentiation solver
* [`notebook/5-2-implicit_diff.ipynb`](notebook/5-2-implicit_diff.ipynb)

### Section 6.1 — Tabular classification
* [`notebook/6-1-tabular_classification.ipynb`](notebook/6-1-tabular_classification.ipynb)
```bash
pip install -e .[classif]
```

### Section 6.2 — Model-Based RL

```bash
pip install -e .[mbrl]
python scripts/mbrl.py --dataset "mujoco/halfcheetah/simple-v0" --n_experiments 5
```
- Implicit diff solver: [`notebook/6-2-implicit_diff_mbrl.ipynb`](notebook/6-2-implicit_diff_mbrl.ipynb)
- Results visualization: [`notebook/6-2-table_mbrl.ipynb`](notebook/6-2-table_mbrl.ipynb)

## ⚖️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## 🤝 Open-source Participation

Do not hesitate to contribute to this project by submitting pull requests or issues, we would be happy to receive feedback and integrate your suggestions.

---

## 📚 Citing

If you use our repository in your work, please cite our paper:

```bibtex
@inproceedings{benechehab2026embedding,
   title={Embedding Distance as a Reward Signal can replace Verifiers for {LLM} Reasoning},
   author={Abdelhakim Benechehab and Youssef Attia El Hili and Albert Thomas and Giuseppe Paolo and Maurizio Filippone},
   booktitle={ICLR 2026 Workshop on Logical Reasoning of Large Language Models},
   year={2026},
   url={https://openreview.net/forum?id=5P1F5sSFjN}
}
@misc{benechehab2025nllpo,
      title={From Data to Rewards: a Bilevel Optimization Perspective on Maximum Likelihood Estimation},
      author={Abdelhakim Benechehab and Gabriel Singer and Corentin Léger and Youssef Attia El Hili and Giuseppe Paolo and Albert Thomas and Maurizio Filippone and Balázs Kégl},
      year={2025},
      eprint={2510.07624},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2510.07624},
}
```
