import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaTokenizer, RobertaModel


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
