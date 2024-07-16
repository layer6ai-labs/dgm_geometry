from dataclasses import dataclass
from typing import Literal

import numpy as np
import skdim
import torch
from joblib import Parallel, delayed
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset

from data.transforms.unpack import UnpackBatch

from ..base import LIDEstimator


def find_closest_points_parallel(x_batch, data, n_jobs=-1):
    # Function to calculate distances from one point to all points in data
    def find_closest_single_point(x):
        # Compute squared Euclidean distance
        distances = np.sum((data - x) ** 2, axis=1)
        # Return the index of the minimum distance
        return np.argmin(distances)

    # Parallel computation of closest indices
    closest_idx = Parallel(n_jobs=n_jobs)(delayed(find_closest_single_point)(x) for x in x_batch)
    return np.array(closest_idx)


class SkdimLIDEstimator(LIDEstimator):
    """
    A wrapper class for all the LID estimation techniques implemented in the skdim library.
    """

    @dataclass
    class Artifact:
        pass

    def __init__(
        self,
        data,
        ambient_dim: int,
        estimator_type: Literal["ESS", "FisherS", "lPCA", "MLE"],
        unpack: UnpackBatch | None = None,
        device: torch.device | None = None,
        **kwargs,
    ):
        super().__init__(
            data=data,
            ambient_dim=ambient_dim,
            device=device,
            unpack=unpack,
        )

        # instantiate the inner estimator
        if estimator_type == "ESS":
            self._estimator = skdim.id.ESS(**kwargs)
        elif estimator_type == "FisherS":
            self._estimator = skdim.id.FisherS(**kwargs)
        elif estimator_type == "lPCA":
            self._estimator = skdim.id.lPCA(**kwargs)
        elif estimator_type == "MLE":
            self._estimator = skdim.id.MLE(**kwargs)
        else:
            raise ValueError(f"Estimator type {estimator_type} is not supported.")

        # change the data to numpy if it is a torch tensor
        self.data = data

    def _get_data_concatenated(self) -> np.ndarray:
        data = self.data
        if isinstance(data, torch.Tensor):
            data_ret = data.cpu().float().numpy()
        elif isinstance(data, TorchDataset):
            assert len(data) > 0, "Dataset is empty."
            # stack all the elements in data in a torch tensor
            dloader = TorchDataLoader(data, batch_size=128, shuffle=False)
            data_ret = []
            for batch in dloader:
                batch = self.unpack(batch)
                data_ret.append(batch)
            data_ret = torch.cat(data_ret).cpu().float().numpy()
        else:
            raise ValueError(f"Data type {type(data)} is not supported.")

        data_ret = data_ret.reshape(data_ret.shape[0], -1)  # flatten

        # make sure self._data is a numpy array
        assert isinstance(data_ret, np.ndarray), "Data should be a numpy array."
        # make sure self._data has dtype float32
        assert data_ret.dtype == np.float32, "Data should have dtype float32."
        assert data_ret.shape[1] == self.ambient_dim, "Ambient dimension does not match the data."
        return data_ret

    def fit(self, n_neighbors: int = 10, n_jobs: int = 1, **kwargs):
        self._processed_data = self._get_data_concatenated()

        if isinstance(self._estimator, skdim._commonfuncs.LocalEstimator):
            self._lid = self._estimator.fit(
                self._processed_data, n_neighbors=n_neighbors, n_jobs=n_jobs, **kwargs
            ).dimension_pw_
        elif isinstance(self._estimator, skdim._commonfuncs.GlobalEstimator):
            self._lid = self._estimator.fit_pw(
                self._processed_data, n_neighbors=n_neighbors, n_jobs=n_jobs, **kwargs
            ).dimension_pw_
        else:
            raise ValueError("Estimator type not recognized.")

    def _estimate_lid(self, x: torch.Tensor | np.ndarray, n_jobs: int = -1):
        """
        For each row of 'x' finds the closest point in the data and returns the LID of that point.
        """
        x = x.flatten(start_dim=1)  # flatten
        # find the closest point to every point in x in the data
        was_torch = False
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            was_torch = True
        idx = find_closest_points_parallel(x, self._processed_data, n_jobs=n_jobs)
        ret = self._lid[idx]
        if was_torch:
            ret = torch.from_numpy(ret)
        return ret
