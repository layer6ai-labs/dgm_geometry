from typing import Tuple

import torch

from .lid_base import LIDDistribution


class Lollipop(LIDDistribution):
    """
    Samples data from a Lollipop distribution.

    A sample is either drawn from the candy, the stick, or the dot depending on
    the probability mass weights assigned to each of these components.

    The location of the candy and stick is also determined by the parameters.
    """

    def __init__(
        self,
        center_loc: Tuple[float, float] = (3.0, 3.0),
        radius: float = 1.0,
        stick_end_loc: Tuple[float, float] = (1.5, 1.5),
        dot_loc: Tuple[float, float] = (0.0, 0.0),
        candy_ratio: float = 4,
        stick_ratio: float = 2,
        dot_ratio: float = 1,
    ):
        """

        Args:
            center_loc (Tuple[float, float], optional): The location of the center of the candy. Defaults to (3.0, 3.0).
            radius (float, optional): The radius of the candy. Defaults to 1.0.
            stick_end_loc (Tuple[float, float], optional): The coordinates of the end of the stick. Defaults to (1.5, 1.5).
            dot_loc (Tuple[float, float], optional): The coordinates of the dot. Defaults to (0.0, 0.0).
            candy_ratio (float, optional): The relative count of the samples that are from the candy. Defaults to 4.
            stick_ratio (float, optional): The relative count of the samples that are from the stick. Defaults to 2.
            dot_ratio (float, optional): The relative count of the samples that are from the dot. Defaults to 1.
        """

        # Store the categorical probabilities
        self.candy_prob = 1.0 * candy_ratio / (candy_ratio + stick_ratio + dot_ratio)
        self.stick_prob = 1.0 * stick_ratio / (candy_ratio + stick_ratio + dot_ratio)
        self.dot_prob = 1.0 * dot_ratio / (candy_ratio + stick_ratio + dot_ratio)
        self.radius = radius

        # store the locations
        self.dot_loc = torch.Tensor(dot_loc)
        self.center_loc = torch.Tensor(center_loc)
        self.stick_end_loc = torch.Tensor(stick_end_loc)

        dist = torch.norm(self.stick_end_loc - self.center_loc).item()
        self.stick_start_loc = (
            radius / dist * self.stick_end_loc + (1 - radius / dist) * self.center_loc
        )  # the place where the stick touches the candy

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
                return a tensor of shape (sample_count, 2) containing the samples and an optional tensor
                of shape (sample_count) containing the intrinsic dimensionalities of the samples.
        """
        with torch.random.fork_rng():
            if isinstance(sample_shape, int):
                sample_shape = (sample_shape, 2)
            # check shape to be N x ambient_dim
            assert len(sample_shape) == 1 or (
                len(sample_shape) == 2 and sample_shape[1] == 2
            ), "Sample shape should be N x 2"
            sample_count = sample_shape[0]
            if seed is not None:
                torch.manual_seed(seed)
            # (1) Generate sample_count uniformly sampled instances from the candy (circle)
            # non-uniform radius sampling using change-of-variables
            radii = torch.sqrt(torch.rand(sample_count)) * self.radius
            # uniform angle sampling
            angles = torch.rand(sample_count) * 2 * torch.pi
            # Calculate the x and y coordinates of the points within the circle
            candy_samples = torch.stack(
                [
                    self.center_loc[0] + radii * torch.cos(angles),
                    self.center_loc[1] + radii * torch.sin(angles),
                ]
            ).T

            # (2) Generate sample_count samples from the stick
            coeff = torch.rand(sample_count).reshape(-1, 1)
            # sample a convex combination of the start and end of the stick
            stick_samples = coeff @ self.stick_start_loc.reshape(1, -1) + (
                1 - coeff
            ) @ self.stick_end_loc.reshape(1, -1)

            # (3) Generate sample_count samples from the dot
            dot_samples = self.dot_loc.unsqueeze(0).repeat(sample_count, 1)

            # get uniform samples in the range of 0 to 1 of size sample_count to decide which sample to pick
            decision = torch.rand(sample_count)
            # get the mask for each of the three types of samples
            msk_candy = decision < self.candy_prob
            msk_stick = (decision >= self.candy_prob) & (
                decision < (self.candy_prob + self.stick_prob)
            )
            msk_dot = decision >= (self.candy_prob + self.stick_prob)

            ret_data = torch.zeros((sample_count, 2))
            ret_data[msk_candy] = candy_samples[msk_candy]
            ret_data[msk_stick] = stick_samples[msk_stick]
            ret_data[msk_dot] = dot_samples[msk_dot]
        if return_dict:
            ret_lid = torch.zeros(sample_count)
            ret_lid[msk_candy] = 2
            ret_lid[msk_stick] = 1
            ret_lid[msk_dot] = 0
            return {
                "samples": ret_data,
                "lid": ret_lid.long(),
                "idx": ret_lid.clone().long(),
            }
        return ret_data
