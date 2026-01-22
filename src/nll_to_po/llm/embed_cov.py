import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from nll_to_po.training.data import generate_r1_prompt_answer


class AnswerEmbeddingDataset(Dataset):
    """Dataset wrapper for answer strings."""

    def __init__(self, answers):
        self.answers = answers

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        return self.answers[idx]


class EmbeddingExtractor(nn.Module):
    """
    Extract embeddings from an AutoModel with various pooling strategies.
    """

    def __init__(
        self,
        model_name: str,
        pooling: str = "mean",
        max_length: int = 2048,
        device: str | None = None,
    ):
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling = pooling
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze model parameters (no training needed)
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

    def _mean_pool(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def _last_token_pool(self, hidden_states, attention_mask):
        idx = attention_mask.sum(dim=1) - 1
        batch = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch, idx]

    def _cls_pool(self, hidden_states):
        return hidden_states[:, 0, :]

    def _pool(self, hidden_states, attention_mask):
        if self.pooling == "mean":
            return self._mean_pool(hidden_states, attention_mask)
        elif self.pooling == "last":
            return self._last_token_pool(hidden_states, attention_mask)
        elif self.pooling == "cls":
            return self._cls_pool(hidden_states)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    @torch.no_grad()
    def encode_batch(self, texts):
        """Encode a batch of texts into embeddings."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        pooled = self._pool(outputs.last_hidden_state, inputs["attention_mask"])

        return pooled


def collate_fn(batch):
    """Simple collate function that returns list of strings."""
    return batch


def compute_answer_embeddings(
    dataset,
    model_name: str,
    pooling: str = "mean",
    batch_size: int = 32,
    max_length: int = 2048,
):
    """
    Compute embeddings for all answers in the dataset.

    Args:
        dataset: HuggingFace dataset with 'answer' field
        model_name: Name of the HuggingFace model to use
        pooling: Pooling strategy ('mean' or 'last')
        batch_size: Batch size for processing
        max_length: Maximum sequence length

    Returns:
        torch.Tensor: (N, D) tensor of embeddings where N is number of samples
                      and D is the embedding dimension
    """
    # Extract answers
    answers = dataset["answer"]

    # Initialize model
    extractor = EmbeddingExtractor(
        model_name=model_name,
        pooling=pooling,
        max_length=max_length,
    )

    # Create dataloader
    answer_dataset = AnswerEmbeddingDataset(answers)
    dataloader = DataLoader(
        answer_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Compute embeddings
    all_embeddings = []

    print(f"Computing embeddings for {len(answers)} answers...")
    for batch in tqdm(dataloader):
        embeddings = extractor.encode_batch(batch)
        all_embeddings.append(embeddings.cpu())

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)

    return all_embeddings


def compute_covariance_trace(embeddings):
    """
    Compute the trace of the covariance matrix of embeddings.

    Args:
        embeddings: (N, D) tensor of embeddings

    Returns:
        float: Trace of the covariance matrix
    """
    # Center the embeddings
    mean = embeddings.mean(dim=0, keepdim=True)
    centered = embeddings - mean

    # Compute covariance matrix: (1/N) * X^T X
    # where X is the centered embeddings matrix
    N = embeddings.shape[0]
    cov = (centered.T @ centered) / N

    # Compute trace
    trace = torch.trace(cov).item()

    return trace, cov


def compute_U_star_from_covariance(
    Sigma: torch.Tensor, lambda_param: float
) -> torch.Tensor:
    """U* = (λ/2) * Σ^{-1}"""
    n = Sigma.shape[0]
    eps = 1e-6
    Sigma_reg = Sigma + eps * torch.eye(n)
    return (lambda_param / 2) * torch.linalg.inv(Sigma_reg)


def compute_U_star_from_trace(
    trace: float, n: int, lambda_param: float
) -> torch.Tensor:
    """U* = (λ*n / (2*Tr(Σ))) * I_n"""
    scalar = (lambda_param * n) / (2 * trace)
    return scalar * torch.eye(n)


def main():
    parser = argparse.ArgumentParser(
        description="Compute embeddings and covariance trace for dataset answers"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name for embeddings",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceTB/Countdown-Task-GOLD",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="verified_Qwen2.5-7B-Instruct",
        help="Dataset configuration/subset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the trace value (JSON format)",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "last", "cls"],
        help="Pooling strategy for embeddings",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding computation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset_name} ({args.dataset_config})")
    dataset = load_dataset(args.dataset_name, args.dataset_config)["train"]
    dataset = dataset.map(lambda x: generate_r1_prompt_answer(x["messages"]))

    # Compute embeddings
    embeddings = compute_answer_embeddings(
        dataset=dataset,
        model_name=args.model_name,
        pooling=args.pooling,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print(f"Embedding shape: {embeddings.shape}")

    # Compute covariance and trace
    print("Computing covariance matrix and trace...")
    trace, cov = compute_covariance_trace(embeddings)

    print(f"Covariance trace: {trace}")

    # Save results
    output_path = (
        Path(args.output_path)
        / f"{args.dataset_name.replace('/', '-')}"
        / args.model_name.split("/")[-1]
    )
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "trace.json"

    results = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "pooling": args.pooling,
        "num_samples": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
        "covariance_trace": trace,
    }
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Trace saved to: {json_path}")

    # embeddings_path = output_path / "embedding.pt"
    # embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    # torch.save(embeddings, embeddings_path)
    # print(f"Embeddings saved to: {embeddings_path}")

    cov_path = output_path / "covariance.pt"
    cov_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cov, cov_path)
    print(f"Covariance matrix saved to: {cov_path}")


if __name__ == "__main__":
    main()
