import torch
from nflows.flows.base import Flow

from data.transforms.perturbations import PerturbBatch
from data.transforms.unpack import UnpackBatch
from models.training import LightningDGM


class LightningFlow(LightningDGM):
    def __init__(
        self,
        normalizing_flow: Flow,
        sampling_transform=None,
        optim_partial=lambda p: torch.optim.AdamW(p, lr=1e-3),
        scheduler_partial=None,
        unpack_batch: UnpackBatch | None = None,
        perturb_batch: PerturbBatch | None = None,
    ):
        super().__init__(
            sampling_transform=sampling_transform,
            optim_partial=optim_partial,
            scheduler_partial=scheduler_partial,
            unpack_batch=unpack_batch,
            perturb_batch=perturb_batch,
        )
        # capture the inner nflows.Flow
        self.nf: Flow = normalizing_flow

    @property
    def dgm(self):
        return self.nf

    def loss(self, batch):
        """A very simple log_prob loss for the normalizing flow."""
        return -self.nf.log_prob(batch).mean()

    @torch.no_grad()
    def sample(
        self,
        num,
        batch_size=128,
        sampling_transform: bool = True,
    ):
        """
        Sample from the normalizing flow model.
        """
        all_samples = []
        for L in range(0, num, batch_size):
            R = min(L + batch_size, num)
            sz = R - L
            samples = self.nf.sample(sz)
            if self.sampling_transform is not None and sampling_transform:
                samples = self.sampling_transform(samples)
            all_samples.append(samples)
        return torch.cat(all_samples, dim=0)
