"""
A parent class of all the lightning training modules that are being used in the repository.

The only thing that one needs to override is the `loss` and `sample` method for new models.
"""

from abc import ABC
from typing import Callable

import lightning as L
import torch

from data.transforms.perturbations import PerturbBatch
from data.transforms.unpack import UnpackBatch


class LightningDGM(L.LightningModule, ABC):
    def __init__(
        self,
        sampling_transform=None,
        optim_partial: Callable = lambda p: torch.optim.AdamW(p, lr=1e-3),
        unpack_batch: UnpackBatch | None = None,
        perturb_batch: PerturbBatch | None = None,
        scheduler_partial=None,
    ):
        super().__init__()
        self.sampling_transform = sampling_transform
        self.optim_partial = optim_partial
        self.scheduler_partial = scheduler_partial

        if unpack_batch is None:
            self.unpack_batch = UnpackBatch()  # the default identity function
        else:
            self.unpack_batch = unpack_batch

        if perturb_batch is None:
            self.perturb_batch = PerturbBatch()  # the default identity function
        else:
            self.perturb_batch = perturb_batch

    def configure_optimizers(self):
        optimizer = self.optim_partial(self.parameters())
        if self.scheduler_partial is not None:
            # TODO: needs fix for schedulers that do not take num_training_steps
            scheduler = self.scheduler_partial(
                optimizer=optimizer,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        batch_unpacked = self.unpack_batch(batch)
        batch_processed = self.perturb_batch(batch_unpacked)
        loss = self.loss(batch_processed)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_unpacked = self.unpack_batch(batch)
        batch_processed = self.perturb_batch(batch_unpacked)
        loss = self.loss(batch_processed)
        self.log("val/loss", loss)
        return loss

    @property
    def dgm(self) -> torch.nn.Module:
        """
        This is supposed to hold a global interface to the actual torch model that is being trained.
        """
        raise NotImplementedError("Please implement the dgm property.")
