import math
import numbers
from typing import Optional
from typing import Tuple
import warnings

import torch
from torch import Tensor
import torch.nn as nn


class URLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(URLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.igates = nn.Linear(input_size, hidden_size * 4)
        self.hgates = nn.Linear(hidden_size, hidden_size * 4)
        self.forget_bias = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
        u = torch.rand(self.hidden_size) * (1 - 2 / self.hidden_size) + 1 / self.hidden_size
        with torch.no_grad():
            self.forget_bias.copy_(-(1 / u - 1).log())

    def forward(
        self, x: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = self.igates(x) + self.hgates(hx)
        forget_gate, refine_gate, cell_gate, out_gate = gates.chunk(4, 1)
        forget_gate = torch.sigmoid(forget_gate + self.forget_bias)
        refine_gate = torch.sigmoid(refine_gate - self.forget_bias)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)
        effective_gate = 2 * refine_gate * forget_gate + (1 - 2 * refine_gate) * forget_gate ** 2
        cy = effective_gate * cx + (1 - effective_gate) * torch.tanh(cell_gate)
        hy = out_gate * torch.tanh(cy)
        return hy, (hy, cy)


class URLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = False,
        dropout: float = 0.0,
    ):
        super(URLSTM, self).__init__()
        self.layers = nn.ModuleList(
            URLSTMCell(input_size, hidden_size) if i == 0 else URLSTMCell(hidden_size, hidden_size)
            for i in range(num_layers)
        )
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        if (
            not isinstance(dropout, numbers.Number)
            or not 0 <= dropout <= 1
            or isinstance(dropout, bool)
        ):
            raise ValueError(
                "dropout should be a number in range [0, 1] "
                "representing the probability of an element being "
                "zeroed"
            )
        if dropout > 0 and num_layers == 1:
            warnings.warn(
                "dropout option adds dropout after all but last recurrent layer, so non-zero "
                f"dropout expects num_layers greater than 1, but got dropout={dropout} and "
                f"num_layers={num_layers}"
            )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if self.batch_first:
            x = x.transpose(0, 1)
        if state is None:
            state = (
                torch.zeros(len(self.layers), x.shape[1], self.hidden_size, device=x.device),
                torch.zeros(len(self.layers), x.shape[1], self.hidden_size, device=x.device),
            )
        out_states = [], []
        for i, layer in enumerate(self.layers):
            outputs = []
            layer_state = state[0][i], state[1][i]
            for xx in x:
                out, layer_state = layer(xx, layer_state)
                if i != len(self.layers) - 1:
                    out = self.dropout(out)
                outputs.append(out)
            outputs = torch.stack(outputs, 0)
            x = outputs
            out_states[0].append(layer_state[0])
            out_states[1].append(layer_state[1])
        out_states = (torch.stack(out_states[0], 0), torch.stack(out_states[1], 0))
        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        return outputs, out_states
