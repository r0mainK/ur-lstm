import math
from typing import Optional
from typing import Tuple

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
    def __init__(self, input_size: int, hidden_size: int, batch_first: bool = False):
        super(URLSTM, self).__init__()
        self.cell = URLSTMCell(input_size, hidden_size)
        self.batch_first = batch_first

    def forward(
        self, x: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if self.batch_first:
            x = x.transpose(0, 1)
        if state is None:
            state = (
                torch.zeros(x.shape[1], self.cell.hidden_size, device=x.device),
                torch.zeros(x.shape[1], self.cell.hidden_size, device=x.device),
            )
        outputs = []
        for xx in x:
            out, state = self.cell(xx, state)
            outputs.append(out)
        outputs = torch.stack(outputs, 0)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        return outputs, state
