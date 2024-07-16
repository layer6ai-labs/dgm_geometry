from typing import Callable, List

import numpy as np
import torch

from data.transforms.perturbations import GaussianConvolution
from models.training.lightning_ensemble import LightningEnsemble


class LightningLIDL(LightningEnsemble):
    """
    This is a LIDL specific ensemble where the batch processing
    simply adds some Gaussian convolution to the data.
    """

    def __init__(
        self,
        lightning_dgm_partial: List[Callable] | Callable,
        sampling_transform: None,
        optim_partial: List[Callable] | Callable = lambda p: torch.optim.AdamW(p, lr=1e-3),
        scheduler_partial: List[Callable] | Callable | None = None,
        deltas: list[float] | None = None,
        delta: float | None = None,
        num_deltas: int | None = None,
    ):
        # Set the deltas hyperparamater precisely equal to the LIDL paper
        if deltas is None:
            if delta is not None:
                if num_deltas is None:
                    deltas = [
                        delta / 2.0,
                        delta / 1.41,
                        delta,
                        delta * 1.41,
                        delta * 2.0,
                    ]
                else:
                    deltas = [x for x in np.geomspace(delta / 2.0, delta * 2.0, num_deltas)]
            else:
                deltas = [
                    0.010000,
                    0.013895,
                    0.019307,
                    0.026827,
                    0.037276,
                    0.051795,
                    0.071969,
                    0.100000,
                ]
        else:
            assert (
                len(deltas) > 1
            ), "The number of deltas should be greater than 1 for LIDL ensemble."

        self.deltas = deltas
        # add the Gaussian convolution batch processing to the dgm_args
        dgm_args = []
        for delta in deltas:
            dgm_args.append({"perturb_batch": GaussianConvolution(delta=delta)})

        self.sampling_transform = sampling_transform

        super().__init__(
            lightning_dgm_partial=lightning_dgm_partial,
            dgm_args=dgm_args,
            sampling_transforms=sampling_transform,
            optim_partial=optim_partial,
            scheduler_partial=scheduler_partial,
        )

        # Sanity check that all of the inner dgms have a log_prob method
        for i, lightning_dgm in enumerate(self.lightning_dgms):
            dgm = lightning_dgm.dgm
            assert hasattr(
                dgm, "log_prob"
            ), f"[DGM_{i}: {type(dgm)}] Does not have a log_prob function!"
