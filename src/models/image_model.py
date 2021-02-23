from torch import Tensor
import torch.nn as nn

from .utils import create_lstm_variant
from .utils import ModelType


class ImageModel(nn.Module):
    def __init__(
        self,
        model_type: ModelType,
        embedding_dim: int,
        image_size: int,
        hidden_dim_lstm: int,
        hidden_dim_relu: int,
        forget_bias: float,
    ):
        super(ImageModel, self).__init__()
        self.lstm = create_lstm_variant(model_type, embedding_dim, hidden_dim_lstm, forget_bias)
        self.intermediate_dim = hidden_dim_lstm * image_size
        self.linear = nn.Linear(self.intermediate_dim, hidden_dim_relu)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(hidden_dim_relu, 10)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.lstm(x)
        x = x.transpose(0, 1).reshape(-1, self.intermediate_dim)
        x = self.linear(x)
        x = self.relu(x)
        return self.out_proj(x)
