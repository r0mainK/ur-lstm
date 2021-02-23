from typing import Optional
from typing import Tuple

from torch import Tensor
import torch.nn as nn

from .utils import create_lstm_variant
from .utils import ModelType


class LanguageModel(nn.Module):
    def __init__(
        self,
        model_type: ModelType,
        vocab_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout_rate: float,
        forget_bias: float,
    ):
        super(LanguageModel, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = nn.Embedding(vocab_dim, embedding_dim)
        self.lstm = create_lstm_variant(model_type, embedding_dim, hidden_dim, forget_bias)
        self.out_proj = nn.Linear(hidden_dim, embedding_dim)
        self.decoder = nn.Linear(embedding_dim, vocab_dim, bias=False)
        self.decoder.weight = nn.Parameter(self.encoder.weight)

    def forward(
        self, x: Tensor, state: Optional[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        x = self.encoder(x)
        x = self.dropout(x)
        x, state = self.lstm(x, state)
        x = self.out_proj(x)
        x = self.decoder(x)
        return x, state
