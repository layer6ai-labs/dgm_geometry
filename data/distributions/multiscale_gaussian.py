from typing import List, Tuple

import torch

from .lid_base import LIDDistribution


class MultiscaleGaussian(LIDDistribution):
    """
    A simple multivariate Gaussian distribution where you can
    control the eigenspectrum of the covariance matrix by setting
    the `eigenvalues` parameter.

    This is to simulate scenarios where there are multiple scales
    at which you can define intrinsic dimensionality upon.
    """

    def __init__(
        self,
        eigenvalues: List[float],
        seed: int | None = 42,
    ):
        with torch.random.fork_rng():
            if seed is not None:
                torch.manual_seed(seed)
            self.eigvals = torch.tensor(eigenvalues)
            self.ambient_dim = len(eigenvalues)
            self.mean_vector = self.ambient_dim * torch.randn(self.ambient_dim)
            # create a random orthogonal matrix
            orthogonal = torch.randn((self.ambient_dim, self.ambient_dim))
            q, _ = torch.linalg.qr(orthogonal)
            self.covariance_matrix = q @ torch.diag(self.eigvals) @ q.T
            self.distr = torch.distributions.MultivariateNormal(
                loc=self.mean_vector, covariance_matrix=self.covariance_matrix
            )

    def sample(
        self,
        sample_shape: Tuple[int, ...] | int,
        return_dict: bool = False,
        seed: int | None = None,
    ):
        """
        Args:
            sample_shape: The shape of the samples being generated
            return_intrinsic_dimensions (bool, optional): Whether or not to return the intrinsic dimensionalities of the sampled points. Defaults to False.

        Returns:
            (torch.Tensor, (torch.Tensor, optional)):
                return a tensor of shape (sample_count, d) containing the samples and an optional tensor
                of shape (sample_count) containing the intrinsic dimensionalities of the samples.
        """
        with torch.random.fork_rng():
            if isinstance(sample_shape, int):
                sample_shape = (sample_shape, self.ambient_dim)
            # check shape to be N x ambient_dim
            assert len(sample_shape) == 1 or (
                len(sample_shape) == 2 and sample_shape[1] == self.ambient_dim
            ), "Sample shape should be N x d"
            sample_count = sample_shape[0]
            if seed is not None:
                torch.manual_seed(seed)
            data = self.distr.sample((sample_count,))
            lid = torch.ones_like(data[:, 0]).long() * (self.eigvals > 1e-3).sum()
            idx = torch.zeros_like(data[:, 0]).long()

        if return_dict:
            return {
                "samples": data,
                "lid": lid.clone().long(),
                "idx": idx.clone().long(),
            }
        return data
