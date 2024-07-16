"""
This file runs a model-free LID estimation method on a dataset.
This includes the closed-form diffusion method, or the kNN methods in scikit-dimensions,
or the ESS method.

It also follows up by computing the LID estimation evaluation metrics and logs them all
onto mlflow. In addition to that, it provides a heatmap visualization of the LID estimates
on the dataset. The data is passed through a UMAP embedder (if it is higher than 2D), then,
the UMAP is used to visualize the LID estimates on the dataset.

For model-based LID, please refer to the documentation. You should actually run the train
function for them and enable the LID monitoring callback for it to work.
"""

import inspect
import time
from pathlib import Path
from typing import Dict
import os

import dotenv
import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

# get the environment variable IS_TESTING
# when testing, the main function will be called from a
# different file, so we need to import the tools module
# via the scripts package
if os.environ.get("IS_TESTING", False):
    from scripts import tools
else:
    import tools


def get_eval(
    evaluation_methods: ListConfig, gt_lid: np.array, pred_lid: np.array, lbl: str
) -> Dict:
    true_id = float(np.mean(gt_lid))
    pred_id = float(np.mean(pred_lid))
    print(f"[{lbl}] True ID: {true_id}, Pred ID: {pred_id}")
    results = {
        "true_id": true_id,
        "pred_id": pred_id,
        "evaluation_metrics": {},
    }
    for eval_method in evaluation_methods:
        eval_metric_name = eval_method["_target_"].split(".")[-1]
        eval_method_fn = instantiate(eval_method)
        error = eval_method_fn(gt_lid, pred_lid)
        # get the function name from eval_method
        for key in eval_method.keys():
            if key != "_target_" and key != "_partial_":
                eval_metric_name += f"_{key}={eval_method[key]}"
        results["evaluation_metrics"][eval_metric_name] = error
        print(f"[{lbl}] Evaluation on {eval_metric_name}: {error}")
    return results


@hydra.main(version_base=None, config_path="../conf/", config_name="model_free_lid")
@tools.MlflowDecorator(
    exclude_attributes=[  # The hydra attributes to remove from the mlflow logging
        "dataset.all_data_transforms",
        "all_data_transforms",
    ],
    out_dir="./outputs",  # The directory where the artifacts are stored
    experiment_name="ModelFreeLID",  # The name of the experiment to be logged on mlflow
)
def main(cfg: DictConfig, artifact_dir: Path):

    # NOTE: Imports relying on the project should be local so that we can use this function for hydra tests as well!
    from data.datasets.lid import LIDDataset
    from lid.base import LIDEstimator, ModelBasedLIDEstimator
    from visualization.scatterplots import visualize_estimates_heatmap

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Create the dataset (all valid datasets are in the data.datasets package)
    print("Setting up the dataset ...")
    dataset = instantiate(cfg.lid_dataset)
    # get the ambient dimension from the LID dataset
    ambient_dim = cfg.dataset.data_dim

    # Add device to the lid_method arguments and instantiate the LID method
    lid_method_partial = instantiate(cfg.lid_method.estimator)
    lid_method: LIDEstimator = lid_method_partial(
        device=device,
        data=dataset,
        ambient_dim=ambient_dim,
    )
    assert isinstance(
        lid_method, LIDEstimator
    ), "The LID method should be an instance of LIDEstimator."
    assert not isinstance(
        lid_method, ModelBasedLIDEstimator
    ), f"This script is for model-free LID estimation, but got {type(lid_method)}."
    # Load model from checkpoint if specified

    print("Preprocessing the LID estimator ...")
    start_time = time.time()
    # check if default_root_dir is in the lid_method.fit signature
    sig = inspect.signature(lid_method.fit)
    fitting_args = cfg.lid_method.get("preprocess_args", {})
    if "default_root_dir" in sig.parameters:
        fitting_args = {**fitting_args, "default_root_dir": artifact_dir}
    lid_method.fit(**fitting_args)
    fitting_time = time.time() - start_time
    # format the fitting time in days, hours, minutes, and seconds and print it
    print(f"Preprocessing time (s): {fitting_time}")

    # Estimate LID
    print("Estimating LID on the dataset ...")
    start_time = time.time()
    # Turn the dataset into a dataloader if specified
    dataloader_kwargs = cfg.lid_method.get("estimation_args", {}).get("dataloader_kwargs", None)
    assert dataloader_kwargs is not None, "dataloader_kwargs are required in estimation_args."
    assert not ("shuffle" in dataloader_kwargs), "Shuffling the dataset is not allowed."
    assert isinstance(dataset, TorchDataset), "Dataloader kwargs are only valid for torch datasets."
    dloader = TorchDataLoader(dataset, shuffle=False, **dataloader_kwargs)
    pred_lid = []
    gt_lid = []

    # remove dataloader args:
    estimation_args = cfg.lid_method.estimation_args
    del estimation_args.dataloader_kwargs
    all_data_visualization = []
    all_submanifolds = []

    estimation_subsample_limit = cfg.lid_method.estimation_args.get("estimation_subsample", None)
    if estimation_subsample_limit is not None:
        del cfg.lid_method.estimation_args.estimation_subsample
    current_idx = 0
    for batch in tqdm(dloader, desc="Estimating LID on batches ..."):
        if isinstance(dataset, LIDDataset):
            _, gt_lid_batch, submanifold = batch
            all_submanifolds.append(submanifold)
        else:
            gt_lid_batch = torch.ones(len(lid_method.unpack(batch)), dtype=torch.long) * -1
        batch_processed = lid_method.unpack(batch)
        all_data_visualization.append(batch_processed.cpu())
        pred_lid_batch = lid_method.estimate_lid(batch, **estimation_args)
        pred_lid.append(pred_lid_batch)
        # ensure gt_lid_batch is of type long
        assert (
            gt_lid_batch.dtype == torch.long or gt_lid_batch.dtype == torch.int32
        ), f"Ground truth LID should be of type integer for dataset."
        gt_lid.append(gt_lid_batch)
        current_idx += len(gt_lid_batch)
        if estimation_subsample_limit is not None and current_idx >= estimation_subsample_limit:
            break
    pred_lid = torch.cat(pred_lid)
    gt_lid = torch.cat(gt_lid)
    lid_method.ground_truth_lid = gt_lid

    estimation_time = time.time() - start_time
    print(f"Estimation time (s): {estimation_time}")

    # Change the estimates to numpy arrays if not already
    if isinstance(pred_lid, torch.Tensor):
        pred_lid = pred_lid.cpu().numpy()
    if isinstance(lid_method.ground_truth_lid, torch.Tensor):
        lid_method.ground_truth_lid = lid_method.ground_truth_lid.cpu().numpy()

    # Evaluate the LID method and add everything to the results
    results = dict(
        ambient_dim=ambient_dim,
        estimation_time=estimation_time,
        preprocessing_time=fitting_time,
        total_time=fitting_time + estimation_time,
        device=str(device),
        evaluation_metrics={},
    )
    evaluation_methods = cfg.get("evaluation_methods", None)
    if evaluation_methods is not None:
        assert (
            lid_method.ground_truth_lid is not None
        ), "Ground truth LID is required for evaluation!"
        results["global"] = get_eval(
            evaluation_methods, lid_method.ground_truth_lid, pred_lid, "global"
        )
    else:
        print("[Warning!] No evaluation methods specified!")

    if isinstance(dataset, LIDDataset):
        # also enter the results for the submanifolds
        all_submanifolds = torch.cat(all_submanifolds).cpu().numpy()
        unique_submanifolds = np.unique(all_submanifolds)
        if len(unique_submanifolds) > 1:
            for submanifold_idx in unique_submanifolds:
                submanifold_mask = all_submanifolds == submanifold_idx
                # print(submanifold_idx, np.sum(submanifold_mask))

                submanifold_gt = lid_method.ground_truth_lid[submanifold_mask]
                submanifold_pred = pred_lid[submanifold_mask]

                results[f"submanifold_{submanifold_idx}"] = get_eval(
                    evaluation_methods,
                    submanifold_gt,
                    submanifold_pred,
                    f"submanifold_{submanifold_idx}",
                )

    # log results in mlflow in the form of key-value pairs in a yaml file named results.yaml
    mlflow.log_text(OmegaConf.to_yaml(results), "results.yaml")

    if cfg.get("visualize_manifold", False):
        print("Storing the UMAP of predictions ...")
        all_data_visualization = torch.cat(all_data_visualization)
        x_visualize = all_data_visualization.flatten(start_dim=1).cpu().numpy()
        min_estimate = max(np.min(pred_lid), 0)
        max_estimate = min(np.max(pred_lid), ambient_dim)
        img_pred, reducer = visualize_estimates_heatmap(
            x_visualize,
            pred_lid,
            "predicted LID",
            return_img=True,
            min_estimate=0 if isinstance(dataset, LIDDataset) else min_estimate,
            max_estimate=(ambient_dim if isinstance(dataset, LIDDataset) else max_estimate),
        )
        mlflow.log_image(img_pred, "lid_image/heatmap_pred.png")
        if isinstance(dataset, LIDDataset):
            print("Storing the UMAP of ground truth ...")
            img_gt, _ = visualize_estimates_heatmap(
                x_visualize,
                lid_method.ground_truth_lid,
                "ground truth LID",
                return_img=True,
                min_estimate=0,
                max_estimate=ambient_dim,
                reducer=reducer,
            )
            mlflow.log_image(img_gt, "lid_image/heatmap_gt.png")

    # create a pandas dataframe with two columns 'ground_truth_lid' and 'predicted_lid'
    # and store it in the artifact directory
    predictions_df = pd.DataFrame(
        {
            "ground_truth_lid": lid_method.ground_truth_lid,
            "predicted_lid": pred_lid,
        }
    )
    predictions_df.to_csv(artifact_dir / "predictions.csv", index=False)
    mlflow.log_artifact(artifact_dir / "predictions.csv")


if __name__ == "__main__":

    tools.setup_root()
    main()
