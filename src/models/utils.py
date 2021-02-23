from enum import Enum
from typing import Optional

import torch
import torch.nn as nn

from .ur_lstm import URLSTM


class ModelType(str, Enum):
    lstm = "lstm"
    u_lstm = "u-lstm"
    r_lstm = "r-lstm"
    ur_lstm = "ur-lstm"


def create_lstm_variant(
    model_type: ModelType, embdedding_dim: int, hidden_dim: int, forget_bias: Optional[float]
) -> nn.Module:
    if model_type is ModelType.ur_lstm or model_type is ModelType.r_lstm:
        lstm = URLSTM(embdedding_dim, hidden_dim)
        if model_type is ModelType.r_lstm:
            lstm.cell.forget_bias.data = torch.zeros(hidden_dim)
            if forget_bias is not None:
                lstm.cell.igates[:hidden_dim] = forget_bias
                lstm.cell.hgates[:hidden_dim] = forget_bias
        return lstm
    lstm = nn.LSTM(embdedding_dim, hidden_dim)
    if model_type is ModelType.u_lstm:
        u = torch.rand(hidden_dim) * (1 - 2 / hidden_dim) + 1 / hidden_dim
        lstm.bias_ih_l0.data[hidden_dim : 2 * hidden_dim] = -(1 / u - 1).log()
        u = torch.rand(hidden_dim) * (1 - 2 / hidden_dim) + 1 / hidden_dim
        lstm.bias_hh_l0.data[hidden_dim : 2 * hidden_dim] = -(1 / u - 1).log()
    elif forget_bias is not None:
        lstm.bias_ih_l0.data[hidden_dim : 2 * hidden_dim] = forget_bias
        lstm.bias_hh_l0.data[hidden_dim : 2 * hidden_dim] = forget_bias
    return lstm
