* v2: added bert reward
* v3: fix bert reward (set minimum reward to -1000)
* v4: bert reward set minimum reward to -100
* v5: bert reward with the same pattern (regex) checks as equation reward / no format reward
* v6: specific config for sft on gsm8k
* v7: add U_star support with embedding reward + beta>0 (KL penalization)
* v8: fix tokenizer issue
* v9: sft on countdown
* v10: investigating n_epochs vs n_steps
* v11: warmup for grpo
* v12: answer only or answer + rationale reward
* v13: save every 50 steps


# GRPO Training Run Tracker (Detailed)

## Qwen3-0.6B

| Embedder | Reward | Seed 1 | Seed 2 | Seed 3 | Status |
|----------|--------|--------|--------|--------|--------|
| google/embeddinggemma-300m | oracle | 20260127-105010 | 20260127-144650 | 20260127-150349 | |
| google/embeddinggemma-300m | id | 20260127-105022 | 20260127-144713 | 20260127-150404 | |
| google/embeddinggemma-300m | u_cov | 20260127-120937 | 20260127-144918 | 20260127-150422 | |
| google/embeddinggemma-300m | u_trace | 20260127-120925 | 20260127-144929 | 20260127-150414 | |

## Qwen3-1.7B

| Embedder | Reward | Seed 1 | Seed 2 | Seed 3 | Status |
|----------|--------|--------|--------|--------|--------|
| google/embeddinggemma-300m | oracle | 20260127-104959 | | | |
| google/embeddinggemma-300m | id | 20260127-115236 | | | |
| google/embeddinggemma-300m | u_cov | | | | |
| google/embeddinggemma-300m | u_trace | | | | |

## Qwen3-4B

| Embedder | Reward | Seed 1 | Seed 2 | Seed 3 | Status |
|----------|--------|--------|--------|--------|--------|
| google/embeddinggemma-300m | oracle | | | | |
| google/embeddinggemma-300m | id | | | | |
| google/embeddinggemma-300m | u_cov | | | | |
| google/embeddinggemma-300m | u_trace | | | | |

## Qwen3-8B

| Embedder | Reward | Seed 1 | Seed 2 | Seed 3 | Status |
|----------|--------|--------|--------|--------|--------|
| google/embeddinggemma-300m | oracle | | | | |
| google/embeddinggemma-300m | id | | | | |
| google/embeddinggemma-300m | u_cov | | | | |
| google/embeddinggemma-300m | u_trace | | | | |




<div align="center">
<h1>From Data to Rewards: a Bi-level Optimization Perspective on Maximum Likelihood Estimation</h1>

[![paper](https://img.shields.io/static/v1?label=arXiv&message=2402.03885&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2510.07624)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/papers/2510.07624)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/license/MIT)
[![Python: 3.10](https://img.shields.io/badge/Python-3.12-blue)]()

</div>

This repository contains the official implementation of the paper:

   >Abdelhakim Benechehab, Gabriel Singer, Corentin Léger, Youssef Attia El Hili, Giuseppe Paolo, Albert Thomas, Maurizio Filippone, Balázs Kégl.
   [From Data to Rewards: a Bi-level Optimization Perspective on Maximum Likelihood Estimation](https://arxiv.org/abs/2510.07624).

### 📝 Abstract:
Generative models form the backbone of modern machine learning, underpinning state-of-the-art systems in text, vision, and multimodal applications. While Maximum Likelihood Estimation has traditionally served as the dominant training paradigm, recent work have highlighted its limitations, particularly in generalization and susceptibility to catastrophic forgetting compared to Reinforcement Learning techniques, such as Policy Gradient methods. However, these approaches depend on explicit reward signals, which are often unavailable in practice, leaving open the fundamental problem of how to align generative models when only high-quality datasets are accessible. In this work, we address this challenge via a Bilevel Optimization framework, where the reward function is treated as the optimization variable of an outer-level problem, while a policy gradient objective defines the inner-level. We then conduct a theoretical analysis of this optimization problem in a tractable setting and extract insights that, as we demonstrate, generalize to applications such as tabular classification and model-based reinforcement learning.

## 📁 Project Structure

```
├── src/
    ├── nll_to_po
        ├── models
            ├── dn_policy.py                # MLP-based stochastic policies
            └── reward_network.py           # Reward parametrizations
        ├── training
            ├── loss.py                     # Loss functions (e.g., NLL, PG)
            ├── reward.py                   # Reward function wrappers
            ├── data.py                     # Data generators
            └── utils.py
├── notebook/                               # Experimental notebooks
└── scripts/                                # Standalone scripts (e.g., mbrl.py)
```

## 🚀 Installation

### Basic Setup

🔹 Create a conda (or micromamba, or venv) environment
```bash
conda create -n nllpo python=3.12
```

🔹 Activate the environment
```bash
conda activate nllpo
```

🔹 Install the base package (in editable mode)
```bash
pip install -e .
```

### Optional Dependencies

Install additional dependencies based on your experimental needs:

🔹 **For developers** (includes pre-commit hooks 🛠️)
```bash
pip install -e .[dev]
pre-commit install
```
This will enable automatic code formatting and linting using Ruff.

🔹 **For Model-Based Reinforcement Learning experiments**
```bash
pip install -e .[mbrl]
```

🔹 **For Classification experiments**
```bash
pip install -e .[classif]
```

🔹 **For Bilevel optimization experiments (implicit and explicit differentiation notebooks)**
```bash
pip install -e .[bilevel]
```


## 🧪 Experiments

This repository contains several experimental notebooks and scripts. Each experiment corresponds to specific sections in the paper:

###  Section 4.2 - Figure 1: Synthetic data experiment
* [`notebook/4-2-fig1-synthetic.ipynb`](notebook/4-2-fig1-synthetic.ipynb)

###  Section 4.2 - Figure 2: Comparison of the learned distributions
* [`notebook/4-2-fig2-distribution.ipynb`](notebook/4-2-fig2-distribution.ipynb)

###  Section 5.2 - Figure 3: Implicit differentiation solver on synthetic data
* [`notebook/5-2-implicit_diff.ipynb`](notebook/5-2-implicit_diff.ipynb)

###  Section 6.1 - Table 1 and 2: Tabular classification
* [`notebook/6-1-tabular_classification.ipynb`](notebook/6-1-tabular_classification.ipynb)

###  Section 6.2 - Table 3: MBRL

#### 🤖 Running the MBRL experiment: [`scripts/mbrl.py`](scripts/mbrl.py)

- Implements the MBRL experimental pipeline
- Requires the `mbrl` optional dependencies

```bash
# Make sure you have installed the mbrl dependencies
pip install -e .[mbrl]

# Run the MBRL script with default hyperparameters
python scripts/mbrl.py --data_proportion 0.1 --learning_rate 0.001 --n_experiments 5 --dataset "mujoco/halfcheetah/simple-v0" --n_updates 400 --batch_size -1 --entropy_weights 1.0

# See all options
python scripts/mbrl.py --help
```
- To run the implicit diff solver on mbrl data: [`notebook/6-2-implicit_diff_mbrl.ipynb`](notebook/6-2-implicit_diff_mbrl.ipynb)
- The optimal reward constant $u^\star_{\text{im}}$ can then be set in a dictionary at the beginning of the file [`scripts/mbrl.py`](scripts/mbrl.py)

```python
IMPLICIT_DIFF_U_VALUES = {
    "mujoco/halfcheetah/medium-v0": 0.08,
    "mujoco/halfcheetah/expert-v0": 0.15,
    "mujoco/halfcheetah/simple-v0": 0.115,
}
```

#### 📈 Results Visualization: [`notebook/6-2-table_mbrl.ipynb`](notebook/6-2-table_mbrl.ipynb)
- Generates Table 3 entries for MBRL results
- Analyzes the output from the `mbrl.py` script
- Run this notebook after completing the MBRL training (set the path to the results stored in a `*.parquet` file)

```python
results_path = "../logs/mbrl_results/results_mujoco_halfcheetah_medium-v0/results_20250924_211418.parquet"
results_df = pd.read_parquet(results_path)
```

### Bonus: Explicit differentiation solver on synthetic data
* [`notebook/explicit_gradient.ipynb`](notebook/explicit_gradient.ipynb)

## ⚖️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## 🤝 Open-source Participation

Do not hesitate to contribute to this project by submitting pull requests or issues, we would be happy to receive feedback and integrate your suggestions.

---

## 📚 Citing

If you use our repository in your work, please cite our paper:

```bibtex
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
