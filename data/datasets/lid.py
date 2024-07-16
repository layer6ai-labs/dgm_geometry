from torch.utils.data import Dataset as TorchDataset


class LIDDataset(TorchDataset):
    """
    An abstraction for all the datasets that can be used for LID estimation functionalities.
    """

    def __init__(self, ambient_dim: int, x, lid, idx, standardize: bool = False):
        self.ambient_dim = ambient_dim
        self.x = x
        self.lid = lid
        self.idx = idx

        assert len(self.x) == len(self.lid), "The length of x and lid should be the same."
        assert len(self.x) == len(self.idx), "The length of x and idx should be the same."

        if standardize:
            # standardize self.x by scaling and shifting with mean and std
            self.x = (self.x - self.x.mean(dim=0)) / self.x.std(dim=0)

    def __getitem__(self, index):
        return self.x[index], self.lid[index], self.idx[index]

    def __len__(self):
        return len(self.x)
