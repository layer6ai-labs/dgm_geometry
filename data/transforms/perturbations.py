"""
This file contains all batch-level perturbation functions that can be applied to the data.

These transforms will be directly applied to the data before the "training step" or "validation step" functions
in the training modules. This is useful for adding noise to the data, or for applying other types of perturbations
to the data that is stochastic.
"""

import torch


class PerturbBatch:
    """
    A simple identity transform and the parent of all batch processing classes.
    """

    def __init__(self):
        pass

    def __call__(self, batch):
        return batch


class GaussianConvolution(PerturbBatch):
    """
    A simple Gaussian convolution transform that adds Gaussian noise to the batch.
    This is used for LIDL-type intrinsic dimension estimators.
    """

    def __init__(
        self,
        delta: float = 0.01,
    ):
        """

        Args:
            delta (float, optional): The standard deviation of the noise being added. Defaults to 0.01.
        """
        self.delta = delta

    def __call__(self, batch):
        return batch + torch.randn_like(batch) * self.delta
