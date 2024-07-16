from typing import Tuple

import torch
import torch.distributions as dist


class LIDDistribution(dist.Distribution):
    """
    Distribution on union of submanifolds with known intrinsic dimensionality.
    """

    def sample(
        self,
        sample_shape: Tuple[int, ...] | int,
        return_dict: bool = False,
        seed: int | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        If return_dict is True, returns a dictionary with the key 'samples' and 'lid' and 'idx'
        which are the samples, the local intrinsic dimensionality of the submanifold associated
        with that data and 'idx' is the index of submanifold.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")
