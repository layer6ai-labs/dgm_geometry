import math
from typing import Tuple

import torch

from .lid_base import LIDDistribution


class Torus(LIDDistribution):
    """
    This is a distribution that samples from a torus centered at the origin
    with a major radius of `major_r` and a minor radius of `minor_r`.

    You can also include the circle itself as a part of the distribution
    by setting `include_circle` to True. This will make it a distribution
    over the mixture of two submanifold: a circle and a torus embedded in 3D.

    You can also control the ratio of the circle to the torus by setting
    the `mix_ratio` parameter. A value of 0.0 will sample only from the torus
    and a value of 1.0 will sample only from the circle.
    """

    def __init__(
        self,
        major_r: float = 1.0,
        minor_r: float = 0.5,
        include_circle: bool = False,
        mix_ratio: float = 0.5,
    ):
        assert 0.0 <= mix_ratio <= 1.0, "Mix ratio should be in the range [0, 1]"

        self.major_r = major_r
        self.minor_r = minor_r
        if not include_circle:
            self.mix_ratio = 0.0
        else:
            self.mix_ratio = mix_ratio

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
                return a tensor of shape (sample_count, 3) containing the samples and an optional tensor
                of shape (sample_count) containing the intrinsic dimensionalities of the samples.
        """
        with torch.random.fork_rng():
            if isinstance(sample_shape, int):
                sample_shape = (sample_shape, 3)
            # check shape to be N x ambient_dim
            assert len(sample_shape) == 1 or (
                len(sample_shape) == 2 and sample_shape[1] == 3
            ), "Sample shape should be N x 2"
            sample_count = sample_shape[0]
            if seed is not None:
                torch.manual_seed(seed)

            # sample a circle of radius self.major_r
            # by sampling in polar coordinates
            # then transform to cartesian
            theta = torch.rand(sample_count) * 2 * math.pi
            cent_x = self.major_r * torch.cos(theta)
            cent_y = self.major_r * torch.sin(theta)

            phi = torch.rand(sample_count) * 2 * math.pi
            x2 = self.minor_r * torch.cos(phi)
            y2 = self.minor_r * torch.sin(phi)

            idx = (torch.rand(sample_count) > self.mix_ratio).long()
            lid = 1 + idx

        outer = torch.stack([cent_x + torch.cos(theta) * x2, cent_y + torch.sin(theta) * x2, y2]).T
        center = torch.stack([cent_x, cent_y, 0.0 * y2]).T
        data = torch.where(lid[:, None] == 1, center, outer)

        if return_dict:
            return {
                "samples": data,
                "lid": lid.clone().long(),
                "idx": idx.clone().long(),
            }
        return data
