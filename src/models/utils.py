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
            with torch.no_grad():
                lstm.cell.forget_bias.zero_()
                if forget_bias is not None:
                    lstm.cell.igates.bias[:hidden_dim].fill_(forget_bias)
                    lstm.cell.hgates.bias[:hidden_dim].zero_()
        return lstm
    lstm = nn.LSTM(embdedding_dim, hidden_dim)
    with torch.no_grad():
        if model_type is ModelType.u_lstm:
            u = torch.rand(hidden_dim) * (1 - 2 / hidden_dim) + 1 / hidden_dim
            lstm.bias_ih_l0[hidden_dim : 2 * hidden_dim].copy_(-(1 / u - 1).log())
            lstm.bias_hh_l0[hidden_dim : 2 * hidden_dim].zero_()
        elif forget_bias is not None:
            lstm.bias_ih_l0[hidden_dim : 2 * hidden_dim].fill_(forget_bias)
            lstm.bias_hh_l0[hidden_dim : 2 * hidden_dim].zero_()
    return lstm
