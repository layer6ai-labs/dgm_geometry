"""
This code contains an ensemble of lightning dgm models. This ensemble is trained
simultaneously on the same data.

The ensemble is used for LIDL-type estimators for example.
"""

import copy
from typing import Callable, List, Literal

import lightning as L
import torch

from .lightning_dgm import LightningDGM


# The order of inheritence is important here.
class LightningEnsemble(L.LightningModule):
    # TODO: Currently, it is storing all the different models at the same time in GPU memory (I think)
    # This is not ideal for large models. We should consider a way to store them on RAM and load them on demand.
    """
    A generic ensemble class that can be used to train multiple LightningDGM models
    simultaneously on the same data with (possibly) different batch data transforms.
    """

    def __init__(
        self,
        lightning_dgm_partial: List[Callable] | Callable,
        dgm_args: List[dict] | dict | None = None,
        sampling_transforms: List | None = None,
        optim_partial: List[Callable] | Callable = lambda p: torch.optim.AdamW(p, lr=1e-3),
        scheduler_partial: List[Callable] | Callable | None = None,
    ):
        super().__init__()

        # (1) determine the self.ensemble_count
        if isinstance(lightning_dgm_partial, list):
            self.ensemble_count = len(lightning_dgm_partial)
        elif isinstance(dgm_args, list):
            self.ensemble_count = len(dgm_args)
        elif isinstance(sampling_transforms, list):
            self.ensemble_count = len(sampling_transforms)
        elif isinstance(optim_partial, list):
            self.ensemble_count = len(optim_partial)
        elif isinstance(scheduler_partial, list):
            self.ensemble_count = len(scheduler_partial)
        else:
            assert False, "At least one of the arguments should be a list for propagation."

        # (2) propagate all the dictionaries across dimensions
        if not isinstance(lightning_dgm_partial, list):
            lightning_dgm_partial = [lightning_dgm_partial] * self.ensemble_count
        else:
            assert (
                len(lightning_dgm_partial) == self.ensemble_count
            ), "The number of lightning_dgm_partial should be equal to the ensemble_count"
        if not isinstance(dgm_args, list):
            dgm_args = [dgm_args] * self.ensemble_count
        else:
            assert (
                len(dgm_args) == self.ensemble_count
            ), "The number of dgm_args should be equal to the ensemble_count"
        if not isinstance(sampling_transforms, list):
            sampling_transforms = [sampling_transforms] * self.ensemble_count
        else:
            assert (
                len(sampling_transforms) == self.ensemble_count
            ), "The number of sampling_transforms should be equal to the ensemble_count"
        if not isinstance(optim_partial, list):
            optim_partial = [optim_partial] * self.ensemble_count
        else:
            assert (
                len(optim_partial) == self.ensemble_count
            ), "The number of optim_partial should be equal to the ensemble_count"
        if not isinstance(scheduler_partial, list):
            scheduler_partial = [scheduler_partial] * self.ensemble_count
        else:
            assert (
                len(scheduler_partial) == self.ensemble_count
            ), "The number of scheduler_partial should be equal to the ensemble_count"

        # (3) instantiate the lightning_dgms
        self.lightning_dgms: List[LightningDGM] = []
        self.optim_partials: List[Callable] = []
        self.scheduler_partials: List[Callable] = []
        _lightning_dgms = []
        for i in range(self.ensemble_count):
            dgm_arg = dgm_args[i] or {}
            sampling_transform = sampling_transforms[i]
            optim_partial_ = optim_partial[i]
            assert callable(optim_partial_), "The optimizer should be a callable."
            lightning_dgm_partial_ = lightning_dgm_partial[i]
            scheduler_partial_ = scheduler_partial[i]
            # instantiate a lightning dgm model but ignore the optimizer,
            # instead the optimizer is handled manually in the training loop
            _lightning_dgms.append(
                lightning_dgm_partial_(
                    **dgm_arg,
                    sampling_transform=sampling_transform,
                )
            )
            # Important: if we don't copy then the same model will be used for all the ensemble members
            _lightning_dgms[-1] = copy.deepcopy(_lightning_dgms[-1])
            self.optim_partials.append(optim_partial_)
            self.scheduler_partials.append(scheduler_partial_)

        self.lightning_dgms = torch.nn.ModuleList(_lightning_dgms)

        # Activate the manual optimization
        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        for lightning_dgm, optim_partial, scheduler_partial in zip(
            self.lightning_dgms, self.optim_partials, self.scheduler_partials
        ):
            optimizers.append(optim_partial(lightning_dgm.dgm.parameters()))
            if scheduler_partial is not None:
                scheduler = scheduler_partial(
                    optimizers[-1],
                    num_training_steps=self.trainer.estimated_stepping_batches,
                )
                schedulers.append(scheduler)
        if len(schedulers) > 0:
            self.has_lr_scheduler = True
            return optimizers, schedulers
        else:
            self.has_lr_scheduler = False
            return optimizers

    def _step(self, batch, batch_idx, mode: Literal["train", "val"]):

        optimizers = self.optimizers()
        if self.has_lr_scheduler:
            schedulers = self.lr_schedulers()

        log_dict = {}
        for i in range(len(self.lightning_dgms)):
            unpacking_scheme = self.lightning_dgms[i].unpack_batch
            perturbation_scheme = self.lightning_dgms[i].perturb_batch

            batch_unpacked = unpacking_scheme(batch)
            batch_processed = perturbation_scheme(batch_unpacked)
            loss = self.lightning_dgms[i].loss(batch_processed)

            if mode == "train":
                oprimizer: torch.optim.Optimizer = optimizers[i]
                if self.has_lr_scheduler:
                    scheduler = schedulers[i]

                oprimizer.zero_grad()
                self.manual_backward(loss)
                oprimizer.step()

                # Step the scheduler if it is the last batch of the epoch
                if self.has_lr_scheduler and self.trainer.is_last_batch:
                    scheduler.step()

            log_dict[f"{mode}/loss_{i}"] = loss
        self.log_dict(log_dict, prog_bar=True)
        avg_loss = sum([v for k, v in log_dict.items() if "loss" in k]) / len(self.lightning_dgms)
        self.log(f"{mode}/loss", avg_loss)

    def training_step(self, batch, batch_idx):
        self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, "val")

    def sample(self, num, sampling_kwargs: List[dict] | dict | None = None):
        # propagate the sampling_kwargs across the ensemble
        sampling_kwargs = sampling_kwargs or {}
        if isinstance(sampling_kwargs, dict):
            sampling_kwargs = [sampling_kwargs] * len(self.lightning_dgms)

        # sample separately from each model
        samples = []
        for lightning_dgm, kwargs in zip(self.lightning_dgms, sampling_kwargs):
            samples.append(lightning_dgm.sample(num, **kwargs))
        return samples
