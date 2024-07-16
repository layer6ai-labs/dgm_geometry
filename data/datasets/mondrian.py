import torch
from torch.utils.data import Dataset as TorchDataset

from data.distributions import Mondrian


class MondrianDataset(TorchDataset):
    def __init__(self, size=2000, seed=0):
        self.size = size
        with torch.random.fork_rng():
            torch.random.manual_seed(seed)
            distribution = Mondrian()
            self.data = distribution.sample((size,))

    def __getitem__(self, index):
        return {"datapoint": self.data[index]}

    def __len__(self):
        return self.size
