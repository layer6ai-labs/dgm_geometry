"""
A list of different diffeomorphisms that can be used as building blocks for normalizing flows.
"""

import functools
import warnings
from typing import Callable, Literal

import torch
from nflows.nn.nets.resnet import ConvResidualNet, ResidualNet
from nflows.transforms import (
    ActNorm,
    AdditiveCouplingTransform,
    AffineCouplingTransform,
    CompositeTransform,
    PiecewiseRationalQuadraticCouplingTransform,
    Transform,
)
from nflows.transforms.coupling import CouplingTransform

from .diffeomorphisms import Diffeomorphism


class NFlowDiffeomorphism(Diffeomorphism, Transform):
    """
    A wrapper around nflows.Flow to make it a diffeomorphism.
    """

    def __init__(
        self,
        dim: int,
        nflow_transform: Transform,
    ):
        super().__init__()
        self.dim = dim
        self._transform = nflow_transform

    def apply_transform(self, x):
        return self._transform.forward(x)[0]

    def forward(self, x, context=None):
        """
        Takes in noise and generates data out of it!
        """
        return self._transform.forward(x, context)

    def inverse(self, x, context=None):
        """
        Takes in data and generates noise out of it!
        """
        return self._transform.inverse(x, context)


class ConfigurableCouplingFlow(NFlowDiffeomorphism):
    """
    A set of consecutive coupling flows that are configurable using a neural network.
    """

    def __init__(
        self,
        dim: int,
        n_transforms: int,
        coupling_partial: Callable[[torch.Tensor, torch.nn.Module], CouplingTransform],
        transform_net_create_fn: Callable[[int], torch.nn.Module],
        include_actnorm: bool = False,
        flip: bool = False,
    ):
        assert dim > 1, "Dimension should be greater than 1 for a coupling transform."
        if include_actnorm:
            warnings.warn(
                "ActNorm is not recommended for use with coupling transforms, especially in data modelling!"
            )

        nflow_transform = []
        for idx in range(n_transforms):
            # apply a RationalQuadraticSpline
            # apply functools.partial onto the constructor of ResidualNet

            # create an alternating mask
            mask = torch.arange(dim) % 2
            mask = (mask * 2 - 1).int()
            if idx % 2 == 0:
                mask = mask * -1

            coupling: torch.nn.Module = coupling_partial(
                mask=mask if not flip else -mask,
                transform_net_create_fn=transform_net_create_fn,
            )

            nflow_transform.append(coupling)

            # apply an ActNorm
            if include_actnorm:
                nflow_transform.append(ActNorm(dim))

        nflow_transform = CompositeTransform(nflow_transform)

        super().__init__(dim, nflow_transform)


class ResNetCouplingFlow(ConfigurableCouplingFlow):
    """
    A coupling flow based on ResNet architecture.
    """

    def __init__(
        self,
        dim: int,
        n_transforms: int,
        n_hidden: int,
        n_blocks: int,
        coupling_partial: Callable,
        activation: Callable = torch.nn.functional.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
        data_type: Literal["image", "tabular"] = "tabular",
        include_actnorm: bool = False,
        flip: bool = False,
    ):
        if data_type == "tabular":
            partial_constructor = functools.partial(
                ResidualNet,
                hidden_features=n_hidden,
                num_blocks=n_blocks,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
            )
        elif data_type == "image":
            partial_constructor = functools.partial(
                ConvResidualNet,
                hidden_channels=n_hidden,
                num_blocks=n_blocks,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
            )
        super().__init__(
            dim=dim,
            n_transforms=n_transforms,
            coupling_partial=coupling_partial,
            include_actnorm=include_actnorm,
            flip=flip,
            transform_net_create_fn=partial_constructor,
        )


class RQNSF(ResNetCouplingFlow):
    """
    A simple diffeomorphism implemented using RationalQuadraticSpline from nflows.
    """

    def __init__(
        self,
        dim: int,
        n_transforms: int,
        n_hidden: int,
        n_blocks: int,
        activation: Callable = torch.nn.functional.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
        data_type: Literal["image", "tabular"] = "tabular",
        tails: Literal["linear", "quadratic"] = "linear",
        num_bins: int = 32,
        tail_bound: float = 10.0,
        include_actnorm: bool = False,
        flip: bool = False,
    ):
        coupling_partial = functools.partial(
            PiecewiseRationalQuadraticCouplingTransform,
            tails=tails,
            num_bins=num_bins,
            tail_bound=tail_bound,
        )
        super().__init__(
            dim=dim,
            n_transforms=n_transforms,
            n_hidden=n_hidden,
            n_blocks=n_blocks,
            coupling_partial=coupling_partial,
            data_type=data_type,
            include_actnorm=include_actnorm,
            flip=flip,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )


class AffineFlow(ResNetCouplingFlow):
    """
    A simple diffeomorphism implemented using a simple flow from nflows.
    """

    def __init__(
        self,
        dim: int,
        n_transforms: int,
        n_hidden: int,
        n_blocks: int,
        data_type: Literal["image", "tabular"] = "tabular",
        include_actnorm: bool = False,
        flip: bool = False,
    ):
        super().__init__(
            dim=dim,
            n_transforms=n_transforms,
            n_hidden=n_hidden,
            n_blocks=n_blocks,
            coupling_partial=AffineCouplingTransform,
            data_type=data_type,
            include_actnorm=include_actnorm,
            flip=flip,
        )


class AdditiveFlow(ResNetCouplingFlow):
    """
    A simple diffeomorphism implemented using a simple flow from nflows.
    """

    def __init__(
        self,
        dim: int,
        n_transforms: int,
        n_hidden: int,
        n_blocks: int,
        data_type: Literal["image", "tabular"] = "tabular",
        include_actnorm: bool = False,
        flip: bool = False,
    ):
        super().__init__(
            dim=dim,
            n_transforms=n_transforms,
            n_hidden=n_hidden,
            n_blocks=n_blocks,
            coupling_partial=AdditiveCouplingTransform,
            data_type=data_type,
            include_actnorm=include_actnorm,
            flip=flip,
        )
