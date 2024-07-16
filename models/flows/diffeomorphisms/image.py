"""
A list of different diffeomorphisms that can be used as building blocks for normalizing flows.
"""

from typing import Callable, List

from nflows.transforms import ActNorm, CompositeTransform, MultiscaleCompositeTransform
from nflows.transforms.conv import OneByOneConvolution
from nflows.transforms.reshape import SqueezeTransform

from .nflows import ConfigurableCouplingFlow, NFlowDiffeomorphism


class MultiscaleImageFlow(NFlowDiffeomorphism):
    """
    This is a multiscale flow that can be configured to have different scales and different coupling transforms.

    This flow has the following architecture:

    - Block: 1x1 Convolution -> Coupling Transform -> ActNorm

    Every scale uses a set of blocks and performs a "squeeze" operation at the end.

    """

    def __init__(
        self,
        H: int,  # the original height of the image
        W: int,  # the original width of the image
        C: int,  # the number of channels in the image
        squeezing_factors: List[int],  # the scales at which to perform the squeezing operation
        n_blocks: int,  # The size of each block, i.e., the number of compositions (1x1 conv -> coupling -> actnorm) that make up a block
        coupling_partial: Callable[
            [int], ConfigurableCouplingFlow
        ],  # The coupling transform to use
    ):
        cur_H = H
        cur_W = W
        cur_C = C
        all_transforms = MultiscaleCompositeTransform(
            num_transforms=len(squeezing_factors),
        )
        for scale_idx, factor in enumerate(squeezing_factors):
            assert (
                cur_H % factor == 0
            ), f"Height {H} should be divisible by the squeezing factors {squeezing_factors}"
            assert (
                cur_H % factor == 0
            ), f"Width {W} should be divisible by the squeezing factors {squeezing_factors}"
            nflow_transforms = []
            nflow_transforms.append(SqueezeTransform(factor=factor))
            # update the current shape of the image after running through transforms
            cur_H = cur_H // factor
            cur_W = cur_W // factor
            cur_C = cur_C * factor * factor
            for block_idx in range(n_blocks):
                # create a coupling flow for each block
                if scale_idx != 0 or block_idx != 0:
                    nflow_transforms.append(ActNorm(features=cur_C))
                nflow_transforms.append(OneByOneConvolution(num_channels=cur_C))
                coupling = coupling_partial(
                    dim=cur_C,
                    n_transforms=1,
                    include_actnorm=False,
                    flip=((scale_idx * n_blocks + block_idx) % 2 == 1),
                )  # add the coupling transform
                nflow_transforms.append(coupling)
            # composite the transforms
            flow_transform = CompositeTransform(nflow_transforms)
            # make it multiscale by splitting along the channel
            all_transforms.add_transform(
                flow_transform, transform_output_shape=(cur_C, cur_H, cur_W)
            )
            cur_C = cur_C // 2  # do the multiscale splitting

        super().__init__(H * W * C, nflow_transform=all_transforms)
