import torch

from data.transforms.perturbations import PerturbBatch
from data.transforms.unpack import UnpackBatch
from models.diffusions.sdes import Sde
from models.training import LightningDGM


class LightningDiffusion(LightningDGM):
    def __init__(
        self,
        sde: Sde,
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
        self.sde: Sde = sde

    @property
    def dgm(self):
        return self.sde

    @property
    def network(self):
        return self.sde.score_net

    def loss(self, batch):
        bs = batch.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.rand((bs,), device=batch.device)

        # Add noise using the SDE
        noisy_batch, eps = self.sde.solve_forward_sde(batch, timesteps, return_eps=True)

        # Predict the score (roughly, opposite of the noise)
        score_pred = self.network(noisy_batch, timesteps)
        loss = (score_pred + eps).square().flatten(start_dim=1).sum(dim=1)

        return loss.mean()

    def sample(
        self,
        num,
        batch_size=128,
        timesteps=1000,
        sample_shape=None,
        stochastic=True,
        sampling_transform: bool = True,
    ):
        samples = []

        with torch.no_grad():
            while len(samples) * batch_size < num:
                iter_batch_size = min(batch_size, num - batch_size * len(samples))
                if sample_shape is not None:
                    noise_shape = (iter_batch_size,) + sample_shape
                else:
                    noise_shape = (iter_batch_size,)
                # TODO: remove Gaussian prior assumption
                noise = torch.randn(noise_shape).to(self.device)
                if stochastic:
                    sample = self.sde.solve_reverse_sde(noise, steps=timesteps)
                else:
                    sample = self.sde.solve_reverse_ode(noise, steps=timesteps)
                samples.append(sample.cpu())

        samples = torch.cat(samples)
        if self.sampling_transform is not None and sampling_transform:
            samples = self.sampling_transform(samples)
        return samples
