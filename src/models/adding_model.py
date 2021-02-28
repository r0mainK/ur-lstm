from torch import Tensor
import torch.nn as nn

from .utils import create_lstm_variant
from .utils import ModelType


class AddingModel(nn.Module):
    def __init__(self, model_type: ModelType, hidden_size: int, forget_bias: float):
        super(AddingModel, self).__init__()
        self.lstm = create_lstm_variant(model_type, 2, hidden_size, forget_bias)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.lstm(x)
        return self.out_proj(x[-1])
