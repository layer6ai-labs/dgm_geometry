"""
A list of different diffeomorphisms that can be used as building blocks for normalizing flows.
"""

import math
from abc import ABC, abstractmethod
from typing import List

import torch


class Diffeomorphism(torch.nn.Module, ABC):
    """
    A diffeomorphism is a function that is differentiable and has a differentiable inverse.
    This is the base class for all diffeomorphisms used here.
    """

    @abstractmethod
    def apply_transform(self, x: torch.Tensor):
        raise NotImplementedError("This method should be implemented in the subclass.")


class Sinusoidal(Diffeomorphism):
    """
    This diffeomorphism has a good condition number associated with it.
    """

    def __init__(
        self,
        dim: int,
        repeat: int = 5,
        seed=111,
        frequency: float = 10.0,
        kappa_control: float = 1e-4,
    ):
        super().__init__()
        self.dim = dim
        torch.manual_seed(seed)
        self.repeat = repeat
        self.kappa_control = kappa_control
        assert (
            0 < self.kappa_control <= 1
        ), "kappa_control should be in range (0, 1], otherwise it does not form a diffeomorphism."
        if self.repeat > 0:
            # create a set of masks and fill them in the buffer
            self.register_buffer("masks", torch.randint(0, 2, (2 * repeat, dim)).bool())

            # store 'repeat' number of random dim x dim orthogonal matrices as a parameter
            self.register_parameter("rotations", torch.nn.Parameter(torch.randn(repeat, dim, dim)))
            # perform a QR decomposition to get the orthogonal matrix
            for i in range(repeat):
                self.rotations.data[i] = torch.linalg.qr(self.rotations.data[i]).Q
            self.frequency = frequency

    def apply_transform(self, x):
        for i in range(self.repeat):
            x = torch.where(
                self.masks[i + i],
                x,
                x + (1 - self.kappa_control) * torch.sin(x * self.frequency) / self.frequency,
            )
            x = torch.where(self.masks[i + i + 1], x, -x)
            x = x @ self.rotations[i]
        return x


class Rotation(Diffeomorphism):
    """
    A simple diffeomorphism that rotates the input by a fixed angle.
    """

    def __init__(self, dim: int, angles: List[float] | None = None):
        super().__init__()
        if angles is not None:
            # create a d by d rotation matrix with the given angles
            # using rodrigues formula
            self.rot = torch.eye(dim)
            angle_index = 0
            for i in range(dim):
                for j in range(i + 1, dim):
                    assert angle_index < len(
                        angles
                    ), f"Not enough angles provided number of angles should be {dim*(dim-1)//2}"
                    angle = angles[angle_index]
                    angle_index += 1
                    rot_ij = torch.eye(dim)
                    rot_ij[i, i] = math.cos(angle)
                    rot_ij[j, j] = math.cos(angle)
                    rot_ij[i, j] = -math.sin(angle)
                    rot_ij[j, i] = math.sin(angle)
                    self.rot = self.rot @ rot_ij

        else:
            # create a random rotation matrix of dimension dim
            random_matrix = torch.randn(dim, dim)
            # perform a QR decomposition to get the orthogonal matrix
            q, _ = torch.qr(random_matrix)
            self.rot = q

    def apply_transform(self, x):
        rotation_cloned = self.rot.clone().detach()
        rotation_cloned = rotation_cloned.to(x.device)
        return x @ rotation_cloned
