"""
This file contains a model-free method for LID estimation using a closed-form diffusion model.
Assuming the data-distribution is a mixture of delta Dirac distributions, we can find a closed
form formula for the marginal probabilities of a diffusion process.

Furthermore, using the marginal probabilities, one can obtain the Gaussian convolutions and use
the derivative of the Gaussian convolution to derive LID.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset

from data.transforms.unpack import UnpackBatch
from lid import LIDEstimator


class CFDM_LID(LIDEstimator):
    @dataclass
    class Artifact:
        d: int
        distances: torch.Tensor

    def _get_data_concatenated(self) -> torch.Tensor:
        data = self.data
        if isinstance(data, torch.Tensor):
            data_ret = data.cpu().float()
        elif isinstance(data, TorchDataset):
            assert len(data) > 0, "Dataset is empty."
            # stack all the elements in data in a torch tensor
            dloader = TorchDataLoader(data, batch_size=128, shuffle=False)
            data_ret = []
            for batch in dloader:
                batch = self.unpack(batch)
                data_ret.append(batch)
            data_ret = torch.cat(data_ret).cpu().float()
        else:
            raise ValueError(f"Data type {type(data)} is not supported.")

        data_ret = data_ret.flatten(start_dim=1)
        assert (
            data_ret.dtype == torch.float32
        ), f"Data should have dtype float32, but got {data_ret.dtype}"
        assert data_ret.shape[1] == self.ambient_dim, "Ambient dimension does not match the data."
        return data_ret

    def __init__(
        self,
        data: torch.Tensor | TorchDataset,
        ambient_dim: int,
        device: torch.device | None = None,
        beta_min: float = 0.1,
        beta_max: float = 20,
        t_max: float = 1.0,
        unpack: UnpackBatch | None = None,
    ):
        super().__init__(
            data,
            ambient_dim,
            unpack=unpack,
        )

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = self._get_data_concatenated()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.t_max = t_max

    def fit(self):
        pass

    def beta_integral(self, t_start, t_end):
        """Integrate beta(t) from t_start to t_end"""
        if not hasattr(self, "beta_diff"):
            self.beta_diff = self.beta_max - self.beta_min
        t_diff = t_end - t_start
        return self.beta_diff / (2 * self.t_max) * (t_end**2 - t_start**2) + self.beta_min * t_diff

    def beta(self, t):
        return (self.beta_max - self.beta_min) * t / self.t_max + self.beta_min

    def sigma(self, t_end, t_start=0):
        """The standard deviation of x(t_end) | x(t_start)"""
        return math.sqrt(1.0 - math.exp(-self.beta_integral(t_start, t_end)))

    def get_tau(self, t):
        beta_integral_t = self.beta_integral(0, t)
        beta_integral_t = min(beta_integral_t, 500)
        return 2 * (math.exp(beta_integral_t) - 1)

    def _preprocess(self, x: torch.Tensor | np.ndarray, **kwargs) -> Artifact:
        x = x.flatten(start_dim=1)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        elif not isinstance(x, torch.Tensor):
            raise ValueError("The input should be a numpy array or a torch tensor.")

        distances = torch.cdist(x.to(self.device), self.data.to(self.device)).cpu()

        return CFDM_LID.Artifact(d=x.numel() // x.shape[0], distances=distances)

    def compute_lid_from_artifact(
        self,
        lid_artifact: Artifact | None = None,
        t: float = 1e-4,
        knn_k: int | None = None,
        distance_threshold: float | None = None,
    ):
        distances = lid_artifact.distances.to(self.device)
        # only keep the top k distances of every row
        if knn_k is not None:
            distances = torch.topk(distances, knn_k, dim=1, largest=False).values
        tau = self.get_tau(t)
        if distance_threshold is not None:
            # replace all the distances that are greater than the threshold with -inf
            msk = distances > distance_threshold
            distances = torch.where(
                distances > distance_threshold,
                torch.full_like(distances, -math.inf),
                distances,
            )

        logits = -distances * distances / tau
        probs = torch.softmax(logits, dim=-1)
        if distance_threshold is not None:
            logits = torch.where(msk, torch.zeros_like(logits), logits)
        return 2 * torch.sum(-probs * logits, dim=-1).cpu()
