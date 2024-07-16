import math
from dataclasses import dataclass

import torch
from einops import rearrange, repeat
from tqdm import tqdm

from data.transforms.unpack import UnpackBatch
from lid import ModelBasedLIDEstimator
from models.diffusions.sdes import Sde


class NormalBundleEstimator(ModelBasedLIDEstimator):
    """The intrinsic dimension estimator described by Stanczuk et al. (2023).

    See the paper (specifically algorithm 1, as of version 5) for details:
    https://arxiv.org/abs/2212.12611.
    Please note: the paper assumes the diffusion model is variance-exploding.
    This version of the code implements the corresponding algorithm without checks.

    Args:
        sde: An Sde object containing a trained diffusion model
        ambient_dim: Corresponds to d in the paper. Inferred by estimate_id if not
            specified here.
    """

    @dataclass
    class Artifact:
        """A class containing the singular values of the normal bundle at each point.
        This can be used in and of itself for further analysis, or to be stored for
        later LID estimation.
        """

        singular_values: torch.Tensor

    def __init__(
        self,
        model: Sde,
        ambient_dim: int | None = None,
        device: torch.device | None = None,
        unpack: UnpackBatch | None = None,
    ):
        super().__init__(
            ambient_dim=ambient_dim,
            model=model,
            device=device,
            unpack=unpack,
        )
        self.sde: Sde = self.model

    @torch.no_grad
    def _preprocess(
        self,
        x: torch.Tensor,
        noise_time=1e-4,
        num_scores=None,
        score_batch_size=128,
        verbose: int = 0,
        use_device_for_svd: bool = True,
    ) -> Artifact:
        """Perform the computation necessary for LID estimation.

        Args:
            x: The points at which to estimate LID with shape (batch_size, *).
            noise_time: A small, positive number representing t_0 in the paper.
            num_scores: The number of score vectors to sample, corresponding to K
                in the paper, and set to 4*self.ambient_dim by default.
            score_batch_size: The maximum number of simultaneous score-vector computations
                to perform; set this according to hardware.
            verbose: If > 0, display a progress bar.
        """
        # assert if x is not a torch tensor
        assert isinstance(x, torch.Tensor), "x should be a torch.Tensor"

        if num_scores is None:
            num_scores = 4 * self.ambient_dim
        noise_time = torch.tensor(noise_time, device=self.device)

        singular_vals = []
        # Loop through each point in the batch
        x_wrapped = (
            tqdm(x, desc=f"Computing {num_scores} scores for {x.shape[0]} points")
            if verbose > 0
            else x
        )
        for x_point in x_wrapped:
            x_repeated = repeat(x_point.cpu(), "... -> ns ...", ns=num_scores)

            # Populate a matrix of scores at noised-out points
            scores = []
            for x_batch in x_repeated.split(score_batch_size):
                x_batch = x_batch.to(self.device)

                # Sample some noised-out points
                x_eps = self.sde.solve_forward_sde(x_batch, noise_time).reshape(x_batch.shape)

                # Compute scores for each sampled point
                x_eps_scores = self.sde.score(x_eps, noise_time)
                # move to CPU if we are planning on performing SVD on CPU
                if not use_device_for_svd:
                    x_eps_scores = x_eps_scores.cpu()
                scores.append(x_eps_scores)

            # Get the singular values of the score to compute the normal space
            score_matrix = rearrange(torch.cat(scores), "s ... -> s (...)")
            singular_vals.append(torch.linalg.svdvals(score_matrix))

        singular_vals = torch.stack(singular_vals)
        return NormalBundleEstimator.Artifact(singular_vals)

    @torch.no_grad
    def compute_lid_from_artifact(
        self,
        lid_artifact: Artifact | None = None,
        singular_value_threshold: float | None = None,
    ) -> torch.Tensor:
        # count the number of singular values that are more than the threshold
        singular_vals = lid_artifact.singular_values.to(self.device)

        if singular_value_threshold is None:
            normal_dim = (singular_vals[:, :-1] - singular_vals[:, 1:]).argmax(dim=1) + 1
            lids = (self.ambient_dim - normal_dim).cpu()
        else:
            threshold = math.exp(-2 * singular_value_threshold)
            lids = (singular_vals < threshold).sum(dim=1).cpu()

        return lids
