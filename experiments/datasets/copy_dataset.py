from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import IterableDataset


class CopyDataset(IterableDataset):
    def __init__(self, time_lag: int):
        super(CopyDataset).__init__()
        self.seq_length = time_lag + 20

    def __iter__(self) -> Iterator[Tensor]:
        while True:
            ids = torch.zeros(self.seq_length, dtype=torch.long)
            ids[:10] = torch.randint(1, 9, (10,))
            ids[-10:] = torch.ones(10) * 9
            x = torch.zeros(self.seq_length, 10)
            x[range(self.seq_length), ids] = 1
            yield x, ids[:10]
