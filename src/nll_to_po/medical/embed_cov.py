import json
from pathlib import Path
from dataclasses import dataclass

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
import tyro

from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from nll_to_po.medical.utils import prepare_dataset


# =========================
# DATASET WRAPPER
# =========================

class AnswerEmbeddingDataset(Dataset):
    def __init__(self, answers):
        self.answers = answers

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        return self.answers[idx]


def collate_fn(batch):
    return batch


# =========================
# EMBEDDING EXTRACTORS
# =========================

class EmbeddingExtractor(torch.nn.Module):
    def __init__(self, model_name, pooling="mean", max_length=2048):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pooling = pooling

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()
        self.max_length = max_length

    def _mean_pool(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1)
        return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def _cls_pool(self, hidden_states):
        return hidden_states[:, 0]

    def _pool(self, hidden_states, attention_mask):
        if self.pooling == "mean":
            return self._mean_pool(hidden_states, attention_mask)
        elif self.pooling == "cls":
            return self._cls_pool(hidden_states)
        else:
            raise ValueError

    @torch.no_grad()
    def encode_batch(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        return self._pool(outputs.last_hidden_state, inputs["attention_mask"])


class SentenceTransformerEmbeddingExtractor(torch.nn.Module):
    def __init__(self, model_name, max_length=2048):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        # self.model.max_seq_length = max_length

    @torch.no_grad()
    def encode_batch(self, texts):
        return self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )


# =========================
# CORE LOGIC
# =========================

def compute_answer_embeddings(
    dataset,
    model_name,
    batch_size,
    max_length,
    pooling,
    is_sentence_transformer,
):
    # 🔥 CHANGE: use long_answer
    answers = dataset["long_answer"]

    if is_sentence_transformer:
        extractor = SentenceTransformerEmbeddingExtractor(model_name, max_length)
    else:
        extractor = EmbeddingExtractor(model_name, pooling, max_length)

    dataloader = DataLoader(
        AnswerEmbeddingDataset(answers),
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    all_embeddings = []

    for batch in tqdm(dataloader):
        emb = extractor.encode_batch(batch)
        all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0)


def compute_covariance_trace(embeddings):
    mean = embeddings.mean(0, keepdim=True)
    centered = embeddings - mean

    N = embeddings.shape[0]
    cov = (centered.T @ centered) / N
    trace = torch.trace(cov).item()

    return trace, cov


# =========================
# TYRO CONFIG
# =========================

@dataclass
class Args:
    model_name: str
    dataset_name: str = "pubmed_qa"
    split: str = "train"

    output_path: str = "./outputs"

    pooling: str = "mean"
    batch_size: int = 32
    max_length: int = 2048

    is_sentence_transformer: bool = True


# =========================
# MAIN
# =========================

def main(args: Args):
    print("Loading PubMedQA dataset...")
    dataset = load_dataset(
        args.dataset_name,
        split=args.split,
    )

    dataset = prepare_dataset(dataset)

    embeddings = compute_answer_embeddings(
        dataset,
        args.model_name,
        args.batch_size,
        args.max_length,
        args.pooling,
        args.is_sentence_transformer,
    )

    print(f"Embeddings shape: {embeddings.shape}")

    trace, cov = compute_covariance_trace(embeddings)
    print(f"Trace: {trace}")

    output_path = (
        Path(args.output_path)
        / f"{args.dataset_name.replace('/', '-')}"
        / args.model_name.split("/")[-1]
    )
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "trace.json", "w") as f:
        json.dump(
            {
                "covariance_trace": trace,
                "num_samples": embeddings.shape[0],
                "embedding_dim": embeddings.shape[1],
                "model_name": args.model_name,
                "dataset_name": args.dataset_name,
                "pooling": args.pooling
            },
            f,
            indent=2,
        )

    torch.save(cov, output_path / "covariance.pt")


if __name__ == "__main__":
    main(tyro.cli(Args))