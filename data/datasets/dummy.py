import torch
from torch.utils.data import Dataset as TorchDataset


class DummyDataset(TorchDataset):
    """Used to make lightning trainer run validation without a dataset"""

    def __init__(self):
        self.items = torch.zeros(1, 1)

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)
