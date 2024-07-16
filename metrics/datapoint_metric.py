from abc import ABC, abstractmethod

import torch
from tqdm import tqdm


class DatapointMetric(ABC):
    """A scalar metric computed on a datapoint with respect to some model

    Examples include out-of-distribution scores or LIDs.
    """

    def __init__(self, model, device=None):
        self.model = model

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    @abstractmethod
    def score_batch(self, batch):
        """Get a score for a batch of data"""
        pass

    def score_dataloader(self, dataloader):
        """Get a score for a full dataloader"""
        self.model.to(self.device)
        scores = []
        for batch in tqdm(dataloader, desc="Getting scores"):
            with torch.no_grad():
                scores.append(self.score_batch(batch).cpu())
        self.model.cpu()
        return torch.cat(scores)
