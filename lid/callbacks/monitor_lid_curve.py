import dataclasses
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.utils
import torchvision.transforms.functional as TVF
from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from data.datasets import LIDDataset
from lid.base import ModelBasedLIDEstimator
from models.monitoring import MonitorMetrics
from models.training import LightningDGM, LightningEnsemble
from visualization.trend import plot_trends


def _parse_artifact(artifact) -> Dict[str, np.array]:
    # artifact is a dataclass, iterate over all of the fields and if they are
    # of type np.array or torch.tensor, then convert them to numpy array
    # then return the dictionary
    parsed_artifact = {}
    for field in dataclasses.fields(artifact):
        value = getattr(artifact, field.name)
        if isinstance(value, torch.Tensor):
            value = value.cpu()
        parsed_artifact[field.name] = value
    return parsed_artifact  # of shape {key: (batch_size, ...), ...}


class MonitorLIDCurve(MonitorMetrics):
    """
    This class is created for monitoring the performance of model-based LID estimators
    as the training progresses. The main idea is to sweep over a hyperparameter of the LID
    estimator and see how the LID estimate changes as we change the hyperparameter. This
    is especially useful for understanding the behavior of the LID estimator and how it
    changes as the model starts to fit the data manifold.

    **The logging scheme**:

    This callback logs all the information required for the LID curve in the mlflow artifact
    directory associated with the lightning module. Note that this callback always picks a
    fixed subsample size of the original dataset to compute the LID curve. This is done to
    avoid memory issues and to make the computation faster. The following files are stored
    as artifacts:

    1. monitoring_{estimator}/trends/sweeping_range.csv:
        This is a csv file containing the sweeping range of the hyperparameter that was
        swept over. The row count is equal to the number of hyperparameters that were swept
        over and the column count is equal to 1 with its label being equal to that hyperparameter.

    2. monitoring_{estimator}/trends/trend_lid_epoch={epoch:04d}.csv:
        This is a csv file containing the LID estimates for each subsample of the data
        for each epoch. The row count is equal to the number of subsamples under consideration
        and the column count is equal to the number of hyperparameters that were swept over.

    3. monitoring_{estimator}/trends/trend_lid_epoch={epoch:04d}.png:
        This is a plot of the LID curve for each subsample of the data for each epoch.
        The x-axis is the hyperparameter that was swept over and the y-axis is the LID estimate.
        The color coding of the curve is based on the submanifold index if the dataset is a LIDDataset.
        If the dataset is not a LIDDataset, then there is a uniform color coding for all the datapoints.

    4. monitoring_{estimator}/samples/preprocess_artifact_epoch={epoch:04d}_{attr_name}.csv/npy:
        Some LID estimators require a preprocessed artifact to compute the LID curve. This
        is stored as a numpy file like this. The artifact is computed once and then used to
        sweep over the hyperparameter. As an example, in NormalBundles, the artifact is the
        singular value decomposition of the normal bundles.

    **LID estimation hyperparameters**:

    1. use_artifact:
        When this is set to true, then computing the LID curve becomes much cheaper. However, some
        LID estimators don't support this feature. As an example, consider the NormalBundles estimator.
        We cna set the use_artifact to True and then the callback will first compute the Normal bundles
        SVD decomposition. Then, it will store that and sweep over the singular_value_threshold hyperparameter
        to quickly compute the LID curve. This is much faster than recomputing the normal bundles at each
        point.
    2. lid_estimator_partial:
        This is a partial function that takes in a model and returns a LID estimate. This is used to instantiate
        the LID estimator. Before training starts, the callback will retrieve the actual DGM model using the
        `dgm` attribute of the lightning module.
    3. sweeping_arg:
        This is the hyperparameter that we are sweeping over. 
    4. sweeping_range:
        This is the range of the hyperparameter that we are sweeping over. 
    5. lid_estimation_args:
        All of the other arguments that are required for the LID estimator. 
    6. lid_preprocessing_args:
        All of the other arguments that are required for the LID estimator preprocessing. For example, in the NormalBundlesEstimator,
        these arguments are passed in for the preprocess function.

    **Examples**:

    1. Use the NormalBundleEstimator to monitor the LID curve. The best sweeping hyperparameter
    here is the singular_value_threshold hyperparameter. This hyperparameter is used to threshold
    the singular values of the normal bundle at each point. The LID curve should plateau at around
    the correct intrinsic dimensionality.

    2. Use the FastFlowLID to monitor the LID curve. The best sweeping hyperparameter
    here is the delta hyperparameter which is precisely the hyperparameter that LIDL uses.
    Again, the LID curve should plateau at around the correct intrinsic dimensionality.

    3. Use the JacobianFlowLIDEstimator to monitor the LID curve. This is also similar to 
    the FastFlowLID estimator but uses the Jacobian of the flow to compute the LID estimate.
    """

    @property
    def _callback_name(self):
        """Set as the class name of the LID estimator associated with it"""
        return f"{self.lid_estimator.__class__.__name__}Curve"  # the name of the LID estimator

    def __init__(
        self,
        dataset: LIDDataset | TorchDataset,
        lid_estimator_partial: Callable[[torch.nn.Module], torch.Tensor],
        sweeping_arg: str,
        sweeping_range: Iterable,
        use_artifact: bool,
        device: torch.device,
        frequency: int | None = 1,
        subsample_size: int | None = None,
        batch_size: int = 128,
        verbose: bool = True,
        lid_estimation_args: Dict[str, Any] | None = None,
        lid_preprocessing_args: Dict[str, Any] | None = None,
        save_image: bool = False,
        custom_logging_name: str | None = None,
        sampling_kwargs: Dict[str, Any] | None = None,
    ):
        """
        The arguments are similar to the base MonitorLID callback with the following additions:

        Args:
            lid_estimator_partial:
                This is a partial function that takes in a model and returns a LID estimate. This is used to instantiate
                the LID estimator. Before training starts, the callback will retrieve the actual DGM model using the
                `dgm` attribute of the lightning module.

                NOTE: this can sometimes be set to None if the LID estimation method does not have an actual model-based
                LID estimator object associated with it. This is just to cover the case with LIDL for now.

            sweeping_arg (str):
                The hyperparameter that we are sweeping over.
            sweeping_range (Iterable):
                The range of the hyperparameter that we are sweeping over.
            use_artifact (bool):
                Whether to use the artifact for LID estimation.
            lid_estimation_args (Dict[str, Any] | None):
                All of the other arguments that are required for the LID estimator.
            lid_preprocessing_args (Dict[str, Any] | None):
                All of the other arguments that are required for the LID estimator preprocessing.
            save_image (bool): Whether to save the image of the subsampled data, only set to true if the data is image.
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
        self.lid_estimator_partial = lid_estimator_partial

        # set the preprocessing and lid estimation arguments and parse them from DictConfig to actual dictionary
        self.lid_estimation_args = {}
        self.lid_preprocessing_args = {}
        for key in (lid_estimation_args or {}).keys():
            self.lid_estimation_args[key] = lid_estimation_args[key]
        for key in (lid_preprocessing_args or {}).keys():
            self.lid_preprocessing_args[key] = lid_preprocessing_args[key]

        # setup the LID curve hyperparameters
        self.use_artifact = use_artifact
        self.sweeping_arg = sweeping_arg
        self.sweeping_range = sweeping_range

    def _on_train_start(self, trainer: Trainer, pl_module: LightningDGM):
        assert not isinstance(
            pl_module, LightningEnsemble
        ), "Ensemble models are not supported with custom LID estimators."

        self.lid_estimator: ModelBasedLIDEstimator = self.lid_estimator_partial(
            model=pl_module.dgm,
            device=self.device,
            unpack=pl_module.unpack_batch,
        )
        if self.verbose:
            print(
                f"[LID Callback {self.callback_name}] Instantiating the model-based LID estimator ..."
            )
        assert isinstance(
            self.lid_estimator, ModelBasedLIDEstimator
        ), f"Invalid Model-based LID estimator {self.lid_estimator}"

        # all the paths
        self.path_plot_fstr = self.callback_name + "/trends/trend_epoch={epoch_num:04d}.png"
        self.path_trend_fstr = self.callback_name + "/trends/trend_epoch={epoch_num:04d}.csv"
        self.lid_preprocess_artifact_fstr = (
            self.callback_name
            + "/trends/trend_epoch={epoch_num:04d}_preprocess_artifact_{attr_name}.{format}"
        )

        os.makedirs(self.artifact_dir / self.callback_name / "trends", exist_ok=True)

        pd.DataFrame({self.sweeping_arg: [x for x in self.sweeping_range]}).to_csv(
            self.artifact_dir / self.callback_name / "trends" / "sweeping_range.csv",
            index=True,
        )

    def _compute_metrics(
        self,
        batch: torch.Tensor,
        trainer: Trainer,
        pl_module: LightningDGM,
        iterator,
    ) -> Dict[str, torch.Tensor]:
        results = {}

        if self.is_lid_dataset:
            _, lid_batch, idx_batch = batch
        else:
            data_batch = batch
            lid_batch = -1 * torch.ones(data_batch.shape[0], device=data_batch.device).long()
            idx_batch = torch.zeros(data_batch.shape[0], device=data_batch.device).long()

        if self.use_artifact:
            artifact = self.lid_estimator.preprocess(
                **{**self.lid_preprocessing_args, "x": batch},
            )
            parsed_artifact = _parse_artifact(artifact)
            for key in parsed_artifact.keys():
                results[key] = parsed_artifact[key]
        self.x_axis = []
        results["lid_trend"] = []
        for i, val in enumerate(self.sweeping_range):
            # check if val is the type of number
            assert isinstance(
                val, (int, float)
            ), f"The sweeping range must be numeric but got {self.sweeping_arg}={type(val)}"
            if self.verbose:
                iterator.set_postfix_str(
                    f"{self.sweeping_arg}={round(val, 3)} [{i+1} / {len(self.sweeping_range)}]"
                )
            self.x_axis.append(val)
            if self.use_artifact:
                lid = self.lid_estimator.compute_lid_from_artifact(
                    **{
                        **self.lid_estimation_args,
                        "lid_artifact": artifact,
                        self.sweeping_arg: val,
                    },
                )
            else:
                lid = self.lid_estimator.estimate_lid(
                    **{
                        **self.lid_estimation_args,
                        "x": batch,
                        self.sweeping_arg: val,
                    },
                )
            results["lid_trend"].append(lid.clone().cpu())
        results["lid_trend"] = (
            torch.stack(results["lid_trend"]).cpu().T
        )  # of shape (batch_size, sweeping_range)

        results["lid"] = lid_batch.cpu()  # of shape (batch_size)
        results["idx"] = idx_batch.cpu()  # of shape (batch_size)
        return results

    def _log_metrics(
        self,
        logging_results: Dict[str, torch.Tensor],
        trainer: Trainer,
        pl_module: LightningDGM,
    ):
        y_axis = logging_results["lid_trend"].cpu().numpy()
        true_lids = logging_results["lid"].cpu().numpy()
        idx = logging_results["idx"].cpu().numpy()

        if self.verbose:
            print(f"[LID Callback {self.callback_name}] Plotting the LID estimation curve ...")

        ## (1) log the LID curve
        len_to_plot = min(self.subsample_size, y_axis.shape[0])
        y_axis = y_axis[:len_to_plot]
        true_lids = true_lids[:len_to_plot]
        idx = idx[:len_to_plot]
        # for each value in idx find the corresponding lid value
        map_idx_to_lid = {i: lid for i, lid in zip(idx, true_lids)}
        labels = []
        for i in range(np.max(idx) + 1):
            assert i in map_idx_to_lid, f"Index {i} not found in the map"
            labels.append(f"Submanifold {i} - LID: {map_idx_to_lid[i]}")
        if not self.is_lid_dataset:
            labels = ["Datapoints"]

        img = plot_trends(
            y_axis=y_axis,
            x_axis=self.x_axis,
            cluster_idx=idx,
            title="LID Estimation Curve",
            labels=labels,
            xlabel=self.sweeping_arg,
            ylabel=f"LID(.;{self.sweeping_arg})",
            alpha=0.1,
        )
        mlflow.log_image(img, self.path_plot_fstr.format(epoch_num=pl_module.current_epoch))

        ## (2) log the LID curve as a csv file
        y_axis = logging_results["lid_trend"].cpu().numpy()  # of shape (N, sweeping_range)
        all_artifacts = {}
        for key in logging_results.keys():
            if key not in ["lid_trend", "lid", "idx"]:
                all_artifacts[key] = [logging_results[key].cpu().numpy()]

        if self.verbose:
            print(f"[LID Callback {self.callback_name}] Saving the LID estimation curve ...")
        len_to_plot = min(self.subsample_size, y_axis.shape[0])
        y_axis = y_axis[:len_to_plot]
        # store y_axis as a numpy array
        pd.DataFrame(y_axis).to_csv(
            self.artifact_dir / self.path_trend_fstr.format(epoch_num=pl_module.current_epoch),
            index=True,
        )

        # (3) log LID artifacts as csv or npy files
        all_artifacts = {key: np.concatenate(all_artifacts[key]) for key in all_artifacts}
        for key, content in all_artifacts.items():
            if len(content.shape) == 2:
                pd.DataFrame(content).to_csv(
                    self.artifact_dir
                    / self.lid_preprocess_artifact_fstr.format(
                        epoch_num=pl_module.current_epoch, attr_name=key, format="csv"
                    ),
                    index=True,
                )
            else:
                np.save(
                    self.artifact_dir
                    / self.lid_preprocess_artifact_fstr.format(
                        epoch_num=pl_module.current_epoch, attr_name=key, format="npy"
                    ),
                    content,
                )
