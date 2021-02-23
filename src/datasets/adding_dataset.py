import random
from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import IterableDataset


class AddingDataset(IterableDataset):
    def __init__(self, seq_length: int):
        super(AddingDataset).__init__()
        self.seq_length = seq_length

    def __iter__(self) -> Iterator[Tensor]:
        while True:
            x = torch.zeros(self.seq_length, 2)
            x[:, 0] = torch.rand(self.seq_length)
            id_1 = random.randint(0, self.seq_length // 2 - 1)
            id_2 = random.randint(self.seq_length // 2 - 1, self.seq_length - 1)
            x[id_1, 1] = 1
            x[id_2, 1] = 1
            yield x, x[id_1, 0] + x[id_2, 0]
