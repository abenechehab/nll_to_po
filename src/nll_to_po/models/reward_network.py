import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaTokenizer, RobertaModel, AutoModel, AutoTokenizer


class RewardMLP(nn.Module):
    """Multi-layer perceptron reward network."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list,
    ):
        super().__init__()
        layers = []
        dims = [2 * input_dim] + hidden_sizes
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

        self.reward = nn.Sequential(
            nn.Linear(dims[-1], 1),
        )

    def forward(self, state):
        """Forward pass to compute mean and standard deviation."""
        common = self.net(state)
        reward = self.reward(common)
        return reward.squeeze(-1)


class RewardMLPMahalanobis(nn.Module):
    """Mahalanobis matrix parametrized reward. Full or diagonal."""

    def __init__(
        self,
        input_dim: int,
        is_diagonal: bool = False,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.matrix = nn.Parameter(
            init_scale * torch.eye(input_dim), requires_grad=True
        )
        if is_diagonal:
            # Register a hook to zero out gradients of off-diagonal elements
            self.matrix.register_hook(
                lambda grad: grad * torch.eye(input_dim, device=grad.device)
            )

    def forward(self, state):
        """Forward pass to compute mean and standard deviation."""
        y, y_hat = state[..., : self.input_dim], state[..., self.input_dim :]
        return -torch.einsum("...i,ij,...j->...", (y - y_hat), self.matrix, (y - y_hat))


class RewardMLPMahalanobisDiag(nn.Module):
    """Single parameter Mahalanobis matrix reward: u * I_n."""

    def __init__(
        self,
        input_dim: int,
        init_param: float = 1.0,
        use_softplus: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.param = nn.Parameter(init_param * torch.ones(1))
        self.use_softplus = use_softplus

    def forward(self, state):
        """Forward pass to compute mean and standard deviation."""
        y, y_hat = state[..., : self.input_dim], state[..., self.input_dim :]
        param = (
            torch.nn.functional.softplus(self.param)
            if self.use_softplus
            else self.param
        )
        matrix = torch.diag(
            param * torch.ones(self.input_dim, device=self.param.device)
        )
        return -torch.einsum("...i,ij,...j->...", (y - y_hat), matrix, (y - y_hat))


class EmbeddingReward(nn.Module):
    """Embedding-based reward function with an MLP encoder."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 8,
        hidden_sizes: list = [64],
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            *[
                layer
                for h_idx in range(len(hidden_sizes) - 1)
                for layer in (
                    nn.Linear(hidden_sizes[h_idx], hidden_sizes[h_idx + 1]),
                    nn.ReLU(),
                )
            ],
            nn.Linear(hidden_sizes[-1], embedding_dim),
        )

    def forward(self, y, yhat):
        e_y = self.encoder(y)
        e_yhat = self.encoder(yhat)
        sim = F.cosine_similarity(e_y, e_yhat, dim=-1)
        return sim


class EmbeddingMahalanobisReward(nn.Module):
    """Embedding-based Mahalanobis reward function with an MLP encoder."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 8,
        hidden_sizes: list = [64],
        train_encoder: bool = True,
        train_matrix: bool = True,
    ):
        super().__init__()
        assert train_encoder or train_matrix, (
            "At least one of train_encoder or train_matrix must be True"
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            *[
                layer
                for h_idx in range(len(hidden_sizes) - 1)
                for layer in (
                    nn.Linear(hidden_sizes[h_idx], hidden_sizes[h_idx + 1]),
                    nn.ReLU(),
                )
            ],
            nn.Linear(hidden_sizes[-1], embedding_dim),
        )
        self.matrix = nn.Parameter(torch.eye(embedding_dim), requires_grad=train_matrix)

        # Freeze encoder parameters if not training
        if not train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, y, yhat):
        e_y = self.encoder(y)
        e_yhat = self.encoder(yhat)
        diff = e_y - e_yhat
        return -torch.einsum("...i,ij,...j->...", diff, self.matrix, diff)


class BertEmbeddingMahalanobisReward(nn.Module):
    """BERT embedding-based Mahalanobis reward function, when y, y_hat are sentences."""

    def __init__(
        self,
        train_encoder: bool = False,  # Freezing BERT's weights
        train_matrix: bool = True,
        max_length: int = 2048,
    ):
        super().__init__()
        # assert train_encoder or train_matrix, (
        #     "At least one of train_encoder or train_matrix must be True"
        # )

        # Load BERT model and tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.bert_model = RobertaModel.from_pretrained("roberta-large")
        self.max_length = max_length

        # Freeze the BERT model if train_encoder is False
        if not train_encoder:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        # Mahalanobis scaling matrix
        self.matrix = nn.Parameter(
            torch.eye(self.bert_model.config.hidden_size), requires_grad=train_matrix
        )

    def encode(self, y, y_hat):
        """Encodes a sentence into its BERT embedding."""
        if isinstance(y, str) and isinstance(y_hat, str):
            y, y_hat = [y], [y_hat]

        assert (
            isinstance(y, list) and isinstance(y_hat, list) and len(y) == len(y_hat)
        ), "Inputs must be lists of strings of same length"

        with torch.no_grad():
            inputs = self.tokenizer(
                y + y_hat,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            outputs = self.bert_model(**inputs)
        # exclude padding tokens
        o = (outputs.last_hidden_state * inputs.attention_mask.unsqueeze(-1)).mean(
            dim=1
        )
        return o[: len(y)], o[len(y) :]

    def forward(self, y, y_hat):
        e_y, e_y_hat = self.encode(y, y_hat)
        diff = e_y - e_y_hat
        return -torch.einsum("...i,ij,...j->...", diff, self.matrix, diff)


class AutoModelEmbeddingMahalanobisReward(nn.Module):
    """
    Embedding-based reward:
        r = - (e(y) - e(y_hat))^T M (e(y) - e(y_hat))

    - Supports multiple pooling strategies
    - PSD-constrained Mahalanobis matrix
    - Works with decoder-only and encoder-decoder LLMs
    """

    def __init__(
        self,
        model_name: str,
        pooling: str = "mean",          # "mean", "last", "attention"
        train_encoder: bool = False,
        train_matrix: bool = True,
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

        if not train_encoder:
            for p in self.model.parameters():
                p.requires_grad = False

        hidden_size = self.model.config.hidden_size

        # PSD Mahalanobis matrix: M = L @ L^T
        self.L = nn.Parameter(
            torch.eye(hidden_size), requires_grad=train_matrix
        )

        # Optional attention pooling
        if pooling == "attention":
            self.attention = nn.Linear(hidden_size, 1, bias=False)

    # ---------------------------
    # Pooling strategies
    # ---------------------------

    def _mean_pool(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def _last_token_pool(self, hidden_states, attention_mask):
        idx = attention_mask.sum(dim=1) - 1
        batch = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch, idx]

    def _attention_pool(self, hidden_states, attention_mask):
        scores = self.attention(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.einsum("bsh,bs->bh", hidden_states, weights)

    def _pool(self, hidden_states, attention_mask):
        if self.pooling == "mean":
            return self._mean_pool(hidden_states, attention_mask)
        elif self.pooling == "last":
            return self._last_token_pool(hidden_states, attention_mask)
        elif self.pooling == "attention":
            return self._attention_pool(hidden_states, attention_mask)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    # ---------------------------
    # Encoding
    # ---------------------------

    def encode(self, y, y_hat):
        if isinstance(y, str):
            y, y_hat = [y], [y_hat]

        inputs = self.tokenizer(
            y + y_hat,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        pooled = self._pool(outputs.last_hidden_state, inputs["attention_mask"])

        return pooled[: len(y)], pooled[len(y):]

    # ---------------------------
    # Reward
    # ---------------------------

    def forward(self, y, y_hat):
        e_y, e_y_hat = self.encode(y, y_hat)
        diff = e_y - e_y_hat

        M = self.L @ self.L.T
        return -torch.einsum("...i,ij,...j->...", diff, M, diff)
