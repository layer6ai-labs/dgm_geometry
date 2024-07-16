import torch
from sklearn.datasets import make_swiss_roll

from .lid_base import LIDDistribution


class SwissRoll(LIDDistribution):
    """
    A wrapper over the swiss roll distribution in scikit-learn.
    """

    def __init__(self, noise: float = 0.0, hole: bool = False):
        self.noise = noise
        self.hole = hole

    def sample(
        self,
        sample_shape,
        return_dict: bool = False,
        seed: int | None = None,
    ):
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape, 3)
        # check shape to be N x ambient_dim
        assert len(sample_shape) == 1 or (
            len(sample_shape) == 2 and sample_shape[1] == 3
        ), "Sample shape should be N x 3"

        n_samples = sample_shape[0]
        x, _ = make_swiss_roll(n_samples, noise=self.noise, random_state=seed)
        # turn x to torch tensor
        x = torch.from_numpy(x).float()
        if return_dict:
            return {
                "samples": x,
                "lid": 2 * torch.ones(n_samples).long(),
                "idx": torch.zeros(n_samples).long(),
            }
        return x
