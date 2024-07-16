import math
import os
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.utils
import torchvision.transforms.functional as TVF
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from data.datasets import LIDDataset
from lid.ensembles import LightningLIDL
from lid.utils import fast_regression
from models.monitoring import MonitorMetrics
from visualization.scatterplots import visualize_estimates_heatmap
from visualization.trend import plot_trends


class MonitorLIDL(MonitorMetrics):
    """
    This Callback gets added onto LightningEnsemble trainers to monitor their approximate likelihoods
    using the LIDL estimator. The premise is that a an ensemble of models trained on different noised
    out scales of the data can hold the intrinsic dimensionality using a linear regression.


    The logging scheme follows that of the MonitorLID callback, but with the following additional
    logging artifacts:

    1. LIDL/likelihoods/deltas.csv:
        These are the list of different deltas (Gaussian noise standard deviations) that
        are used to perturb the data.

    2. LIDL/likelihoods/likelihood_trend={epoch:04d}.csv:
        This shows the likelihood values of all the subsamples for each of the single models
        in the ensemble. A linear regression model is then fitted to these trends to compute
        the actual LID. The rows of this csv are the number of subsamples and the columns are
        the number of models in the ensemble.

    3. LIDL/likelihoods/likelihood_trend={epoch:04d}.png:
        This is a visualization of the csv that is stored above. It shows the likelihood values
        of all the subsamples for each of the single models in the ensemble.

    4. LIDL/predictions/estimates_{epoch:04d}.csv
        This is a csv file containing the LID estimates for all the subsamples. This will output
        the regression coefficient obtained from all of the datapoints.

    5. LIDL/predictions/estimates_{epoch:04d}_umap.png
        This shows a UMAP heatmap of the LID estimates for all the subsamples. This is useful for
        visualizing the LID estimates.


    **LID estimation hyperparameters**:

    1. likelihood_estimation_args:
        This is a list of dictionaries containing all the arguments that are used for computing the likelihood for each
        of the models in the ensemble. If a single dictionary is given, then this dictionary is populated across for
        all models.
    """

    def __init__(
        self,
        dataset: LIDDataset | TorchDataset,
        device: torch.device,
        frequency: int | None = 1,
        subsample_size: int | None = None,
        batch_size: int = 128,
        verbose: bool = True,
        likelihood_estimation_args: List[Dict[str, Any]] | Dict[str, Any] | None = None,
        save_image: bool = False,
        custom_logging_name: str | None = None,
        sampling_kwargs: Dict[str, Any] | None = None,
    ):
        """
        Everything is identical with the parent except for the likelihood_estimation_args
        parameter. This parameter is a list of dictionaries containing all the arguments that
        are used for computing the likelihood for each of the models in the ensemble. If a single
        dictionary is given, then this dictionary is populated across for all models.
        """

        super().__init__(
            dataset=dataset,
            device=device,
            frequency=frequency,
            subsample_size=subsample_size,
            batch_size=batch_size,
            verbose=verbose,
            save_image=save_image,
            custom_logging_name=custom_logging_name,
            sampling_kwargs=sampling_kwargs,
        )

        # parse the likelihood estimation args and turn it into an actual dictionary
        self.likelihood_estimation_args = {}
        for key in (likelihood_estimation_args or {}).keys():
            self.likelihood_estimation_args[key] = likelihood_estimation_args[key]

    @property
    def _callback_name(self):
        """
        Add in to ensure the file naming is correct!
        """
        return "LIDL"

    def _on_train_start(self, trainer: Trainer, pl_module: LightningLIDL) -> None:
        assert isinstance(
            pl_module, LightningLIDL
        ), f"Invalid lightning module it should be an LightningLIDL, but got {type(pl_module)}"

        # setup likelihood estimation arguments
        if isinstance(self.likelihood_estimation_args, list):
            assert len(self.likelihood_estimation_args) == len(
                pl_module.lightning_dgms
            ), f"Invalid likelihood estimation args length"
        else:
            self.likelihood_estimation_args = [self.likelihood_estimation_args] * len(
                pl_module.lightning_dgms
            )

        # all the paths
        self.path_plot_fstr = (
            self.callback_name + "/likelihoods/likelihood_trend={epoch_num:04d}.png"
        )
        self.path_trend_fstr = (
            self.callback_name + "/likelihoods/likelihood_trend={epoch_num:04d}.csv"
        )
        self.path_predictions_fstr = (
            self.callback_name + "/predictions/estimates_{epoch_num:04d}.csv"
        )
        self.path_umap_fstr = self.callback_name + "/predictions/estimates_{epoch_num:04d}_umap.png"

        os.makedirs(self.artifact_dir / self.callback_name / "likelihoods", exist_ok=True)
        os.makedirs(self.artifact_dir / self.callback_name / "predictions", exist_ok=True)

        # store all the delta values
        pd.DataFrame({"delta": [x for x in pl_module.deltas]}).to_csv(
            self.artifact_dir / self.callback_name / "likelihoods" / "deltas.csv",
            index=True,
        )

        # check that the underlying model has a log_prob method which is the foundation of the LIDL estimator
        for lightning_dgm in pl_module.lightning_dgms:
            assert hasattr(
                lightning_dgm.dgm, "log_prob"
            ), f"Invalid model ({lightning_dgm.dgm.__class__.__name__}), it should have a log_prob method"

    def _compute_metrics(
        self, batch: torch.Tensor, trainer: Trainer, pl_module: LightningLIDL, iterator
    ) -> Dict[str, torch.Tensor]:
        if self.is_lid_dataset:
            _, lid_batch, idx_batch = batch
        else:
            data_batch = batch
            lid_batch = -1 * torch.ones(data_batch.shape[0], device=data_batch.device).long()
            idx_batch = torch.zeros(data_batch.shape[0], device=data_batch.device).long()

        self.x_axis = []
        all_likelihoods = []
        for i, delta in enumerate(pl_module.deltas):
            if self.verbose:
                iterator.set_postfix_str(
                    f"delta={round(delta, 3)} [{i+1} / {len(pl_module.deltas)}]"
                )
            self.x_axis.append(math.log(delta))
            batch_unpacked = pl_module.lightning_dgms[i].unpack_batch(batch)
            likelihoods = (
                pl_module.lightning_dgms[i]
                .dgm.log_prob(
                    batch_unpacked.to(self.device),
                    **self.likelihood_estimation_args[i],
                )
                .clone()
            )
            self.ambient_dim = batch_unpacked.numel() // batch_unpacked.shape[0]
            all_likelihoods.append(likelihoods.cpu())

        return {
            "likelihoods": torch.stack(all_likelihoods)
            .cpu()
            .T,  # of shape (batch_size, number of deltas)
            "lid": lid_batch.cpu(),  # of shape (batch_size)
            "idx": idx_batch.cpu(),  # of shape (batch_size)
            "data_flattened": batch_unpacked.flatten(
                start_dim=1
            ).cpu(),  # of shape (batch_size x ambient_dim)
        }

    def _get_predictions(
        self,
        likelihoods,
    ):
        """
        Perform a regression on the likelihoods on the GPU (if available) to compute the LID estimates
        """
        return (
            fast_regression(
                all_ys=torch.tensor(likelihoods, device=self.device),
                xs=torch.tensor(self.x_axis, device=self.device),
            )
            .cpu()
            .numpy()
        ) + self.ambient_dim

    def _log_metrics(
        self,
        logging_results: Dict[str, torch.Tensor],
        trainer: Trainer,
        pl_module: LightningLIDL,
    ):
        if self.verbose:
            print("[LIDL Callback] Plotting likelihood estimation curves ...")
        likelihoods = logging_results["likelihoods"].numpy()  # of shape (data_size x delta_size)
        true_lids = logging_results["lid"].numpy()  # of shape (data_size)
        idx = logging_results["idx"].numpy()  # of shape (data_size)
        data_flattened = logging_results[
            "data_flattened"
        ].numpy()  # of shape (data_size x ambient_dim) we're going to need it for UMAP

        # plot the likelihood curve
        len_to_plot = min(self.subsample_size, likelihoods.shape[0])
        likelihoods = likelihoods[:len_to_plot]
        true_lids = true_lids[:len_to_plot]
        idx = idx[:len_to_plot]
        # for each value in idx find the corresponding lid value
        map_idx_to_lid = {i: lid for i, lid in zip(idx, true_lids)}
        labels = []

        for i in range(np.max(idx).item() + 1):
            assert i in map_idx_to_lid, f"Index {i} not found in the map"
            labels.append(f"Submanifold {i} - LID: {map_idx_to_lid[i]}")
        if not self.is_lid_dataset:
            labels = ["Datapoints"]

        img = plot_trends(
            y_axis=likelihoods,
            x_axis=self.x_axis,
            cluster_idx=idx,
            title="LID Estimation Curve",
            labels=labels,
            xlabel="$\\log \\delta$",
            ylabel=f"log convolution densities",
            alpha=0.1,
        )
        mlflow.log_image(img, self.path_plot_fstr.format(epoch_num=pl_module.current_epoch))

        likelihoods = logging_results["likelihoods"].numpy()  # of shape (data_size x delta_size)

        if self.verbose:
            print("[LIDL Callback] Saving the likelihood estimates ...")

        len_to_plot = min(self.subsample_size, likelihoods.shape[0])
        likelihoods = likelihoods[:len_to_plot]
        # store y_axis as a numpy array
        pd.DataFrame(likelihoods).to_csv(
            self.artifact_dir / self.path_trend_fstr.format(epoch_num=pl_module.current_epoch),
            index=True,
        )

        lid_values = self._get_predictions(likelihoods=likelihoods)

        pd.DataFrame(lid_values).to_csv(
            self.artifact_dir
            / self.path_predictions_fstr.format(epoch_num=pl_module.current_epoch),
            index=True,
        )

        if hasattr(self, "reducer"):
            reducer = self.reducer
            if self.verbose:
                print("[LIDL Callback] Reusing the UMAP reducer ...")
        else:
            if self.verbose:
                print("[LIDL Callback] Fitting UMAP reducer ...")
            reducer = None

        if self.verbose:
            print("[LIDL Callback] Plotting UMAP ...")
        img_pred, reducer = visualize_estimates_heatmap(
            data_flattened,
            lid_values,
            "predicted LID",
            return_img=True,
            min_estimate=0,
            max_estimate=self.ambient_dim,
            reducer=reducer,
        )
        self.reducer = reducer  # set the reducer
        # store the image
        mlflow.log_image(img_pred, self.path_umap_fstr.format(epoch_num=pl_module.current_epoch))
