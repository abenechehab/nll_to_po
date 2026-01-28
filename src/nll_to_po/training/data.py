import re
from typing import Optional, Dict, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch

from sklearn.datasets import load_wine, load_breast_cancer, load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from ucimlrepo import fetch_ucirepo
import minari

from nll_to_po.training.reward import equation_reward_func


UCI_NAME_TO_ID = {"credit_default": 350, "spambase": 94, "poker": 158}

SYSTEM_PROMPT_ORCA = """Solve the given high school math problem by providing a clear explanation of each step leading to the final solution.

Provide a detailed breakdown of your calculations, beginning with an explanation of the problem and describing how you derive each formula, value, or conclusion. Use logical steps that build upon one another, to arrive at the final answer in a systematic manner.

# Steps

1. **Understand the Problem**: Restate the given math problem and clearly identify the main question and any important given values.

2. **Set Up**: Identify the key formulas or concepts that could help solve the problem (e.g., algebraic manipulation, geometry formulas, trigonometric identities).

3. **Solve Step-by-Step**: Iteratively progress through each step of the math problem, justifying why each consecutive operation brings you closer to the solution.

4. **Double Check**: If applicable, double check the work for accuracy and sense, and mention potential alternative approaches if any.

5. **Final Answer**: Provide the numerical or algebraic solution clearly, accompanied by appropriate units if relevant.

# Notes

- Always clearly define any variable or term used.

- Wherever applicable, include unit conversions or context to explain why each formula or step has been chosen.

- Assume the level of mathematics is suitable for high school, and avoid overly advanced math techniques unless they are common at that level.
"""


def _ensure_numpy_X_y(X, y):
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.to_numpy().ravel()

    if y.dtype.kind not in "iu":
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    return X, y.astype(np.int64, copy=False)


def load_uci(
    dataset="wine",
    test_size=0.2,
    val_size=0.2,
    batch_size=256,
    standardize=True,
    impute_missing=False,
    impute_strategy="median",
    random_state=0,
    uci_id=None,
):
    if dataset in {"wine", "iris", "breast_cancer", "load_digits"} and uci_id is None:
        if dataset == "wine":
            data = load_wine()
        elif dataset == "iris":
            data = load_iris()
        elif dataset == "breast_cancer":
            data = load_breast_cancer()
        elif dataset == "load_digits":
            data = load_digits()
        X, y = data.data, data.target

    else:
        if uci_id is None:
            if dataset in UCI_NAME_TO_ID:
                uci_id = UCI_NAME_TO_ID[dataset]
            else:
                raise ValueError(
                    f"Unknown dataset '{dataset}'. "
                    f"Use one of {{'wine','iris','breast_cancer','load_digits'}} "
                    f"or provide a valid UCI id via `uci_id`."
                )
        ds = fetch_ucirepo(id=uci_id)
        X = ds.data.features
        y = ds.data.targets
        if isinstance(y, pd.DataFrame) and y.shape[1] > 1:
            y = y.iloc[:, 0]

        X, y = _ensure_numpy_X_y(X, y)

    if impute_missing:
        imp = SimpleImputer(strategy=impute_strategy)
        X = imp.fit_transform(X)

    if standardize:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)

    X_tr, X_tt, y_tr, y_tt = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=val_size, stratify=y_tr, random_state=random_state
    )

    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_tt = torch.tensor(X_tt, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_tt = torch.tensor(y_tt, dtype=torch.long)

    tr_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, y_tr, y_tr, torch.zeros_like(y_tr)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val, y_val, torch.zeros_like(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )
    tst_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tt, y_tt, y_tt, torch.zeros_like(y_tt)),
        batch_size=batch_size,
        shuffle=False,
    )

    meta = {"input_dim": X.shape[1], "num_classes": int(np.unique(y).size)}
    return tr_loader, val_loader, tst_loader, meta


def generate_data_linear(
    input_dim: int,
    output_dim: int,
    train_size: int = 100,
    val_size: int = 100,
    test_size: int = 100,
    init_dist_loc: Optional[float] = None,
    init_dist_scale: Optional[float] = None,
    init_dist_n_samples: int = 1,
    A: Optional[torch.Tensor] = None,
):
    assert input_dim == output_dim, (
        f"input dim {input_dim} is different from output dim {output_dim}"
    )

    # resample parameters
    if A is None:
        A = torch.eye(output_dim)
    if init_dist_loc is None:
        init_dist_loc = np.random.uniform(-6.0, 6.0)
    if init_dist_scale is None:
        init_dist_scale = np.random.uniform(0.1, 2.5)

    # Generate input data (uniform in -5,5)
    X_unique = torch.rand((train_size + val_size + test_size, input_dim)) * 10 - 5

    # For training data: repeat each unique X multiple times
    X_train_unique = X_unique[:train_size]
    X_train = X_train_unique.repeat_interleave(init_dist_n_samples, dim=0)

    # Compute mean for training X and sample multiple y values
    mean_y_train = X_train_unique @ A.T + init_dist_loc
    mean_y_train_expanded = mean_y_train.repeat_interleave(init_dist_n_samples, dim=0)
    y_train = (
        mean_y_train_expanded
        + torch.randn(X_train.shape[0], output_dim) * init_dist_scale
    )
    std_y_train = torch.full_like(mean_y_train_expanded, init_dist_scale)

    # For validation data: use unique X without repetition
    X_val = X_unique[train_size : train_size + val_size]
    mean_y_val = X_val @ A.T + init_dist_loc
    y_val = mean_y_val + torch.randn(X_val.shape[0], output_dim) * init_dist_scale
    std_y_val = torch.full_like(mean_y_val, init_dist_scale)

    # For test data: use unique X without repetition
    X_test = X_unique[train_size + val_size :]
    mean_y_test = X_test @ A.T + init_dist_loc
    y_test = mean_y_test + torch.randn(X_test.shape[0], output_dim) * init_dist_scale
    std_y_test = torch.full_like(mean_y_test, init_dist_scale)

    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(
        X_train, y_train, mean_y_train_expanded, std_y_train
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=X_train.shape[0], shuffle=True
    )
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val, mean_y_val, std_y_val)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=X_val.shape[0], shuffle=False
    )
    test_dataset = torch.utils.data.TensorDataset(
        X_test, y_test, mean_y_test, std_y_test
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=X_test.shape[0], shuffle=False
    )

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        {
            "init_dist_loc": init_dist_loc,
            "init_dist_scale": init_dist_scale,
            "init_dist_n_samples": init_dist_n_samples,
        },
    )


def generate_data_linear_noloader(
    input_dim: int,
    output_dim: int,
    train_size: int = 100,
    val_size: int = 100,
    test_size: int = 100,
    init_dist_loc: Optional[float] = None,
    init_dist_scale: Optional[float] = None,
    init_dist_n_samples: int = 1,
    A: Optional[torch.Tensor] = None,
):
    assert input_dim == output_dim, (
        f"input dim {input_dim} is different from output dim {output_dim}"
    )

    # resample parameters
    if A is None:
        A = torch.eye(output_dim)
    if init_dist_loc is None:
        init_dist_loc = np.random.uniform(-6.0, 6.0)
    if init_dist_scale is None:
        init_dist_scale = np.random.uniform(0.1, 2.5)

    # Generate input data (uniform in -5,5)
    X_unique = torch.rand((train_size + val_size + test_size, input_dim)) * 10 - 5

    # For training data: repeat each unique X multiple times
    X_train_unique = X_unique[:train_size]
    X_train = X_train_unique.repeat_interleave(init_dist_n_samples, dim=0)

    # Compute mean for training X and sample multiple y values
    mean_y_train = X_train_unique @ A.T + init_dist_loc
    mean_y_train_expanded = mean_y_train.repeat_interleave(init_dist_n_samples, dim=0)
    y_train = (
        mean_y_train_expanded
        + torch.randn(X_train.shape[0], output_dim) * init_dist_scale
    )

    # For validation data: use unique X without repetition
    X_val = X_unique[train_size : train_size + val_size]
    mean_y_val = X_val @ A.T + init_dist_loc
    y_val = mean_y_val + torch.randn(X_val.shape[0], output_dim) * init_dist_scale

    # For test data: use unique X without repetition
    X_test = X_unique[train_size + val_size :]
    mean_y_test = X_test @ A.T + init_dist_loc
    y_test = mean_y_test + torch.randn(X_test.shape[0], output_dim) * init_dist_scale

    return X_train, y_train, X_val, y_val, X_test, y_test


def generate_data_single_point(
    input_dim: int,
    output_dim: int,
    init_dist_loc: Optional[float] = None,
    init_dist_scale: Optional[float] = None,
    init_dist_n_samples: Optional[int] = None,
):
    # resample parameters
    if not init_dist_loc:
        init_dist_loc = np.random.uniform(-5.0, 5.0)
    if not init_dist_scale:
        init_dist_scale = np.random.uniform(0.5, 1.5)
    if not init_dist_n_samples:
        init_dist_n_samples = np.random.randint(1, 100)

    # Generate new random data for each experiment
    X = torch.randn(1, input_dim)
    mean_y = torch.ones((1, output_dim)) * init_dist_loc
    mean_y_expanded = mean_y.repeat_interleave(init_dist_n_samples, dim=0)
    y = mean_y_expanded + torch.randn(init_dist_n_samples, output_dim) * init_dist_scale
    X = X.repeat(init_dist_n_samples, 1)  # Repeat X for each sample
    batch_size = X.shape[0]
    std_y = torch.full_like(mean_y_expanded, init_dist_scale)

    # Create a DataLoader
    train_dataset = torch.utils.data.TensorDataset(X, y, mean_y_expanded, std_y)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    return (
        train_dataloader,
        None,
        None,
        {
            "init_dist_loc": init_dist_loc,
            "init_dist_scale": init_dist_scale,
            "init_dist_n_samples": init_dist_n_samples,
        },
    )


def generate_data_minari_noloader(
    dataset_name: str,
    train_size: float = 0.8,
    data_proportion: float = 1.0,
    batch_size: int = -1,
    test_size: float = 0.1,
    seed: int = 7,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    Dict[str, int],
]:
    """Create train/val/test DataLoaders from a Minari dataset.

    Returns: (train_loader, val_loader, test_loader, {input_dim, output_dim})
    """
    assert train_size + test_size <= 1.0, (
        "train_size + test_size must be <= 1.0 (for validation)"
    )

    dataset = minari.load_dataset(dataset_name, download=True)
    dataset.set_seed(seed=int(seed))

    observations = []
    actions = []
    next_observations = []
    for episode in dataset:
        observations.append(episode.observations[:-1])
        actions.append(episode.actions)
        next_observations.append(episode.observations[1:])
    observations = np.concatenate(observations, axis=0)
    actions = np.concatenate(actions, axis=0)
    next_observations = np.concatenate(next_observations, axis=0)

    obs_dim = observations.shape[1]
    action_dim = actions.shape[1]

    X = torch.tensor(
        np.concatenate([observations, actions], axis=1), dtype=torch.float32
    )
    y = torch.tensor(next_observations, dtype=torch.float32)

    # Shuffle and select subset
    total_size = len(X)
    indices = torch.randperm(total_size)
    selected_size = int(total_size * data_proportion)
    selected_indices = indices[:selected_size]

    X = X[selected_indices]
    y = y[selected_indices]

    # Train/val/test split
    trn_size = int(len(X) * train_size)
    test_size_actual = int(len(X) * test_size)
    val_size = len(X) - trn_size - test_size_actual

    X_train, X_val, X_test = (
        X[:trn_size],
        X[trn_size : trn_size + val_size],
        X[trn_size + val_size :],
    )
    y_train, y_val, y_test = (
        y[:trn_size],
        y[trn_size : trn_size + val_size],
        y[trn_size + val_size :],
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        {
            "input_dim": obs_dim + action_dim,
            "output_dim": obs_dim,
            "dataset_name": "_".join(dataset_name.split("/")),
            "train_size": train_size,
            "data_proportion": data_proportion,
            "batch_size": batch_size,
        },
    )


def generate_data_minari(
    dataset_name: str,
    train_size: float = 0.8,
    data_proportion: float = 1.0,
    batch_size: int = -1,
    test_size: float = 0.1,
    seed: int = 7,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    Dict[str, int],
]:
    """Create train/val/test DataLoaders from a Minari dataset.

    Returns: (train_loader, val_loader, test_loader, {input_dim, output_dim})
    """
    assert train_size + test_size <= 1.0, (
        "train_size + test_size must be <= 1.0 (for validation)"
    )

    seed = np.random.randint(0, 1_000_000)

    dataset = minari.load_dataset(dataset_name, download=True)
    dataset.set_seed(seed=int(seed))

    observations = []
    actions = []
    next_observations = []
    for episode in dataset:
        observations.append(episode.observations[:-1])
        actions.append(episode.actions)
        next_observations.append(episode.observations[1:])
    observations = np.concatenate(observations, axis=0)
    actions = np.concatenate(actions, axis=0)
    next_observations = np.concatenate(next_observations, axis=0)

    obs_dim = observations.shape[1]
    action_dim = actions.shape[1]

    X = torch.tensor(
        np.concatenate([observations, actions], axis=1), dtype=torch.float32
    )
    y = torch.tensor(next_observations, dtype=torch.float32)

    # Shuffle and select subset
    total_size = len(X)
    indices = torch.randperm(total_size)
    selected_size = int(total_size * data_proportion)
    selected_indices = indices[:selected_size]

    X = X[selected_indices]
    y = y[selected_indices]

    # Train/val/test split
    trn_size = int(len(X) * train_size)
    test_size_actual = int(len(X) * test_size)
    val_size = len(X) - trn_size - test_size_actual

    X_train, X_val, X_test = (
        X[:trn_size],
        X[trn_size : trn_size + val_size],
        X[trn_size + val_size :],
    )
    y_train, y_val, y_test = (
        y[:trn_size],
        y[trn_size : trn_size + val_size],
        y[trn_size + val_size :],
    )

    # using y as mu and sigma zero
    train_dataset = torch.utils.data.TensorDataset(
        X_train, y_train, y_train, torch.zeros_like(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size if batch_size > 0 else X_train.shape[0],
        shuffle=True,
    )
    val_dataset = torch.utils.data.TensorDataset(
        X_val, y_val, y_val, torch.zeros_like(y_val)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size if batch_size > 0 else X_val.shape[0],
        shuffle=False,
    )
    test_dataset = torch.utils.data.TensorDataset(
        X_test, y_test, y_test, torch.zeros_like(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size if batch_size > 0 else X_test.shape[0],
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        {
            "input_dim": obs_dim + action_dim,
            "output_dim": obs_dim,
            "dataset_name": "_".join(dataset_name.split("/")),
            "train_size": train_size,
            "data_proportion": data_proportion,
            "batch_size": batch_size,
        },
    )


def generate_r1_prompt(target, numbers):
    r1_prefix = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer. ",
        },
        {
            "role": "user",
            "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>.",
        },
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    return {
        "prompt": r1_prefix,
        "target": target,
    }


def generate_r1_prompt_answer(messages, tokenizer: Optional[object] = None):
    r1_prefix = [
        {
            "role": "system",
            "content": messages[0]["content"],
        },
        {
            "role": "user",
            "content": messages[1]["content"],
        },
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    search_result = re.search(r"<answer>\s*(.*?)\s*</answer>", messages[2]["content"])
    answer = ""
    if search_result is not None:
        answer = search_result.group(1).strip()
    if tokenizer is not None:
        r1_prefix = tokenizer.apply_chat_template(
            r1_prefix, tokenize=False, add_generation_prompt=False
        )
    return {
        "prompt": r1_prefix,
        "trace": messages[2]["content"],
        "answer": answer,
    }


def tokenize(prompt, field_name, tokenizer):
    return {
        field_name: tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        ),
    }


def evaluate(dataset, model, tokenizer, subset_size: int = 100):
    dataset = dataset.shuffle(seed=42).select(range(subset_size))
    dataset = dataset.map(lambda x: generate_r1_prompt_answer(x["messages"]))
    print(f"example prompt before chat template: {dataset[0]['prompt']}")
    dataset = dataset.map(lambda x: tokenize(x["prompt"], "prompt", tokenizer))

    correct = 0
    total = len(dataset)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    visited = False

    for example in tqdm(dataset, desc="Evaluating training accuracy"):
        prompt = example["prompt"]

        # Model prediction
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=0.0,
            )

        pred = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        ).strip()

        # True reward function
        reward = equation_reward_func(
            completions=[pred],
            target=[example["target"]],
            nums=[example["nums"]],
            verbose=int(not visited),
        )
        correct += reward[0]

        if not visited:
            print("\nExample prediction:")
            print(f"Prompt: {prompt}")
            print(f"Predicted answer: {pred}")
            print(f"target: {example['target']}")
            print(f"numbers: {example['nums']}")
            print(f"True answer: {example['answer']}")
            visited = True

    training_accuracy = correct / total

    print("\n==============================")
    print(f"Training accuracy: {training_accuracy:.4f}")
    print(f"Correct: {correct} / {total}")
    print("==============================")
