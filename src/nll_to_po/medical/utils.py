import re
from typing import Any
from datasets import Dataset

import torch

import nll_to_po.models.reward_network as RM


SYSTEM_PROMPT = (
    "You are a medical expert. Given a clinical research abstract, "
    "answer the yes/no question and provide a one-sentence explanation."
)


def format_prompt(sample: dict) -> str:
    contexts = "\n\n".join(
        f"[{label}]\n{context}"
        for label, context in zip(sample["LABELS"], sample["CONTEXTS"])
    )

    prompt = f"""<question>
{sample["QUESTION"]}
</question>

<abstract>
{contexts}
</abstract>

Answer the question with exactly "yes" or "no", then provide a one-sentence explanation of your reasoning.

Respond in this exact format:
<answer>yes/no</answer>
<long_answer>One sentence explanation here.</long_answer>"""

    return prompt


def extract_answer(text: str) -> str | None:
    match = re.search(r"<answer>\s*(yes|no)\s*</answer>", text, re.IGNORECASE)
    return match.group(1).strip().lower() if match else None


def extract_long_answer(text: str) -> str | None:
    match = re.search(r"<long_answer>(.*?)</long_answer>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def answer_correctness_reward(
    completions: list[list[dict[str, str]]],
    answer: list[str],
    **kwargs: Any,
) -> list[float]:
    """Returns 1.0 if the extracted <answer> matches the ground truth, else 0.0."""
    rewards = []
    for completion, ground_truth in zip(completions, answer):
        text = completion[0]["content"]
        predicted = extract_answer(text)
        reward = (
            1.0
            if (predicted is not None and predicted == ground_truth.strip().lower())
            else 0.0
        )
        rewards.append(reward)
    return rewards


def format_compliance_reward(
    completions: list[list[dict[str, str]]],
    **kwargs: Any,
) -> list[float]:
    """Returns 1.0 if both <answer> and <long_answer> tags are present and non-empty, else 0.0."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        has_answer = extract_answer(text) is not None
        has_long_answer = extract_long_answer(text) is not None
        rewards.append(1.0 if (has_answer and has_long_answer) else 0.0)
    return rewards


def combined_reward(
    completions: list[list[dict[str, str]]],
    answers: list[str],
    **kwargs: Any,
) -> list[float]:
    """
    Weighted combination:
      - 0.7 for correct answer
      - 0.3 for format compliance (both tags present)
    """
    rewards = []
    for completion, ground_truth in zip(completions, answers):
        text = completion[0]["content"]
        predicted = extract_answer(text)
        has_long_answer = extract_long_answer(text) is not None

        answer_score = (
            1.0
            if (predicted is not None and predicted == ground_truth.strip().lower())
            else 0.0
        )
        format_score = 1.0 if (predicted is not None and has_long_answer) else 0.0

        rewards.append(0.7 * answer_score + 0.3 * format_score)
    return rewards


def prepare_dataset(hf_dataset) -> Dataset:
    """Map raw HuggingFace dataset rows to prompt/answer pairs for GRPO."""

    def process(sample):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_prompt(sample)},
            ],
            "answer": sample["final_decision"],  # ground truth for reward
            "long_answer": sample["LONG_ANSWER"],  # available for logging/eval
        }

    return hf_dataset.map(process, remove_columns=hf_dataset.column_names)


def embedding_reward_func_constructor_pubmedqa(
    model: str,
    U_star=None,
    pooling: str = "mean",
    verbose: int = 0,
    max_length: int = 2048,
    answer_only: bool = True,
    sentence_transformer: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if sentence_transformer:
        embedder = RM.SentenceTransformerMahalanobisReward(
            model_name=model,
            train_encoder=False,
            train_matrix=False,
            max_length=max_length,
        )
        min_reward = -1e6
    else:
        embedder = RM.AutoModelEmbeddingMahalanobisReward(
            model_name=model,
            train_encoder=False,
            train_matrix=False,
            max_length=max_length,
            pooling=pooling,
        )
        min_reward = -1e6

    if U_star is not None:
        embedder.set_matrix(matrix=torch.nn.Parameter(U_star))
    embedder.to(device)

    def reward_func(
        completions: list[list[dict[str, str]]],
        answer: list[str],  # final_decision ground truth  ("yes" / "no")
        long_answer: list[str],  # LONG_ANSWER ground truth (one-sentence explanation)
        **kwargs,
    ) -> list[float]:
        rewards = []
        for completion, a, la in zip(completions, answer, long_answer):
            try:
                text = completion[0]["content"]

                # ── format gate ──────────────────────────────────────────────
                predicted_answer = extract_answer(text)  # "yes" / "no" or None
                predicted_long = extract_long_answer(text)  # str or None

                if predicted_answer is None:
                    rewards.append(min_reward)
                    continue

                if answer_only:
                    # Embed only the short answer token against the ground-truth decision
                    y_hat = predicted_answer  # e.g. "no"
                    y = a.strip().lower()  # e.g. "no"
                else:
                    # Embed the full explanation against the ground-truth long answer.
                    # Fall back to min_reward if the long_answer tag is missing.
                    if predicted_long is None:
                        rewards.append(min_reward)
                        continue
                    y_hat = predicted_long  # model's one-sentence explanation
                    y = la.strip()  # LONG_ANSWER ground truth

                reward = embedder(y_hat=y_hat, y=y)
                rewards.append(reward.item())

            except Exception as e:
                if verbose > 0:
                    print(
                        f"completion: {completion}\nanswer: {a}\nlong_answer: {la}\nError: {e}"
                    )
                rewards.append(min_reward)

        return rewards

    return reward_func


# -------------------------
# SFT
# -------------------------


def format_sft_sample(sample: dict) -> dict:
    contexts = "\n\n".join(
        f"[{label}]\n{context}"
        for label, context in zip(sample["LABELS"], sample["CONTEXTS"])
    )
    user_content = f"""<question>
{sample["QUESTION"]}
</question>

<abstract>
{contexts}
</abstract>

Answer with "yes" or "no", then provide a one-sentence explanation.

Respond in this exact format:
<answer>yes/no</answer>
<long_answer>One sentence explanation here.</long_answer>"""

    # The <think> block is kept minimal — the model already knows how to think,
    # we only need it to learn to emit <long_answer> after </answer>.
    assistant_content = (
        f"<think>\n{sample['LONG_ANSWER']}\n</think>\n"
        f"<answer>{sample['final_decision']}</answer>\n"
        f"<long_answer>{sample['LONG_ANSWER']}</long_answer>"
    )

    return {
        # prompt-completion conversational format (recommended by TRL docs)
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "completion": [
            {"role": "assistant", "content": assistant_content},
        ],
    }


def prepare_sft_dataset(hf_dataset, num_samples: int | None = None):
    if num_samples is not None:
        hf_dataset = hf_dataset.select(range(num_samples))
    hf_dataset = hf_dataset.filter(
        lambda x: x["final_decision"] is not None and x["LONG_ANSWER"] is not None
    )
    return hf_dataset.map(
        format_sft_sample,
        remove_columns=hf_dataset.column_names,
    )
