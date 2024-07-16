"""
Contains the different visualization utilities used in jupyter notebooks.
"""

from typing import Callable, Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from lid.base import LIDEstimator
from models.diffusions.sdes import Sde


def plot_lid_on_a_grid(
    data,
    lid_estimator: LIDEstimator,
    mode: Literal["with_preprocessing", "without_preprocessing"],
    argument_name: str,
    argument_values: Iterable,
    **other_kwargs,  # the other kwargs that will be passed to the lid estimator
):
    fig, axes = plt.subplots(4, 4, figsize=(16, 13))
    # fig, axes = plt.subplots(1, 2, figsize=(16, 16))

    assert len(argument_values) == len(axes.flatten())
    if mode == "with_preprocessing":
        artifact = lid_estimator.preprocess(data)
    for ax, arg_val in tqdm(
        zip(axes.flatten(), argument_values),
        desc="computing scatterplot",
        total=len(argument_values),
    ):  # Generate 1k points and plot them

        if mode == "with_preprocessing":
            all_lid = lid_estimator.compute_lid_from_artifact(
                artifact,
                **{argument_name: arg_val},
                **other_kwargs,
            ).cpu()
        elif mode == "without_preprocessing":
            all_lid = lid_estimator.estimate_lid(
                data, **{argument_name: arg_val}, **other_kwargs
            ).cpu()
        else:
            raise ValueError("Invalid mode")

        # clip LID values
        all_lid = np.clip(all_lid, 0, lid_estimator.ambient_dim)

        s = ax.scatter(*data.cpu().T, c=all_lid, cmap="plasma", vmin=0, vmax=2)

        ax.set_title(f"{argument_name}={round(arg_val, 3)}")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(s, cax=cax, orientation="vertical")

        cbar.set_label(f"$LID({{{round(arg_val, 3)}}})(\\cdot)$", rotation=90, labelpad=15)

    fig.tight_layout()
    plt.show()


def plot_log_prob_on_a_grid(
    data: torch.Tensor,
    argument_name: str,
    argument_values: Iterable,
    sde: Sde | None = None,
    log_prob_fn: Callable | None = None,
    **log_prob_kwargs,  # log prob kwargs
):
    fig, axes = plt.subplots(4, 4, figsize=(16, 13))
    assert len(argument_values) == len(axes.flatten())

    if log_prob_fn is None:
        assert sde is not None, "sde must be provided if log_prob_fn is not provided"
        log_prob_fn = sde.log_prob
    else:
        assert sde is None, "sde must not be provided if log_prob_fn is provided"

    epsilon_cnt = 0
    for ax, arg_val in tqdm(zip(axes.flatten(), argument_values), total=len(argument_values)):
        epsilon_cnt += 1
        all_log_probs = log_prob_fn(x=data, **{argument_name: arg_val}, **log_prob_kwargs)
        # turn all_log_probs into their ranks

        heatmap = torch.exp(all_log_probs.cpu())
        # Graph the norms
        s = ax.scatter(*data.T.cpu().numpy(), c=heatmap, cmap="plasma")

        ax.set_title(f"{argument_name} = {arg_val}")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(s, cax=cax, orientation="vertical")

        # Add a label to the colorbar
        cbar.set_label("$p(\\cdot)$", rotation=90, labelpad=15)
    fig.tight_layout()


def plot_lid_trend_simple(
    data,
    lid_estimator: LIDEstimator,
    mode: Literal["with_preprocessing", "without_preprocessing"],
    argument_name: str,
    argument_values: Iterable,
    **other_kwargs,  # the other kwargs that will be passed to the lid estimator
):
    lid = []
    x_axis = []

    if mode == "with_preprocessing":
        artifact = lid_estimator.preprocess(data)
    for arg_val in tqdm(argument_values):
        if mode == "with_preprocessing":
            all_lid = lid_estimator.compute_lid_from_artifact(
                artifact,
                **{argument_name: arg_val},
                **other_kwargs,
            ).cpu()
        elif mode == "without_preprocessing":
            all_lid = lid_estimator.estimate_lid(
                data, **{argument_name: arg_val}, **other_kwargs
            ).cpu()
        else:
            raise ValueError("Invalid mode")

        lid.append(all_lid)
        x_axis.append(arg_val)

    lid = torch.stack(lid).T.cpu().numpy()
    x_axis = np.array(x_axis)

    for i in range(len(lid)):
        plt.plot(x_axis, lid[i], alpha=0.1)
    # take the average lid estimate
    avg_lid = lid.mean(axis=0)
    plt.plot(x_axis, avg_lid, color="red")

    plt.title(f"LID estimates")
    plt.xlabel(argument_name)
    plt.ylabel(f"$LID(\\cdot; {argument_name})$")

    plt.show()
