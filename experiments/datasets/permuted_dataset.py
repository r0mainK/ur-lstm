import torch
from torch import Tensor
from torch.utils.data import Dataset


class PermutatedDataset(Dataset):
    def __init__(self, dataset: Dataset, seq_length: int):
        super(PermutatedDataset).__init__()
        self.dataset = dataset
        self.permutations = torch.zeros(len(dataset), seq_length, dtype=torch.long)
        for i in range(len(dataset)):
            self.permutations[i] = torch.randperm(seq_length)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> Tensor:
        x, target = self.dataset[index]
        return x[self.permutations[index]], target
