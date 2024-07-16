from typing import List, Optional, Tuple

import numpy as np
import torch
import umap
import umap.umap_ as umap
from matplotlib import pyplot as plt
from PIL import Image
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from .pretty import ColorTheme, FONT_FAMILY, hashlines, StyleDecorator, savable


def visualize_estimates_heatmap(
    data: np.ndarray,  # [data_size, dim]
    estimates: np.ndarray,  # [data_size]
    title: str,
    max_estimate: float | None = None,
    min_estimate: float | None = None,
    return_img: bool = False,
    alpha: float = 0.1,
    reducer: umap.UMAP | None = None,
):
    """
    This is used when we want to plot an estimand that is assigned
    to all the points of the data.

    What this function does is that it takes all the estimated values
    and then performs a umap embedding (if needed) on data, then
    provides a heatmap where the intensity of a point represents the value
    of the estimand.

    To visualize LID values, for example, we can use this method.
    """
    # train a UMAP embedding on all the data
    if data.shape[1] > 2:
        title = f"{title} (UMAP projection)"
        if reducer is None:
            reducer = umap.UMAP()
            reducer.fit(data)
        embedding = reducer.transform(data)
    else:
        embedding = data
    try:
        s = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            alpha=alpha,
            cmap="plasma",
            c=estimates,
            vmin=min_estimate,
            vmax=max_estimate,
        )
        plt.title(title)
        # fix the colorbar to a
        plt.colorbar(s, orientation="vertical")

    finally:

        img = None
        if not return_img:
            plt.show()
        else:
            fig = plt.gcf()
            fig.canvas.draw()
            np_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                fig.canvas.get_width_height()[::-1] + (3,)
            )
            # create a PIL image out of the np_array of size (H x W x 3)
            img = Image.fromarray(np_array)
        plt.close()

    return img, reducer


@savable
@StyleDecorator(font_scale=2, style="whitegrid", line_style="--")
def pretty_visualize_estimates_heatmap(
    data: np.ndarray,  # [data_size, dim]
    estimates: np.ndarray,  # [data_size]
    max_estimate: float | None = None,
    min_estimate: float | None = None,
    alpha: float = 0.1,
    reducer: umap.UMAP | None = None,
    colorbar_label: str | None = None,
    figsize: Optional[tuple] = (7, 6),
    fontsize: Optional[int] = None,
    no_legend: bool = False,
    legend_fontsize: Optional[int] = None,
    legend_loc: Optional[str] = None,
    title: str | None = None,
    custom_xticks: Optional[int] = None,
    custom_yticks: Optional[int] = None,
    custom_zticks: Optional[int] = None,
    box_ratios: Optional[List[int]] = None,
    remove_ticks_x_label: bool = False,
    remove_ticks_y_label: bool = False,
    remove_ticks_z_label: bool = False,
    cbar_ticks: List[float] | None = None,
    xlim: Tuple | None = None,
    ylim: Tuple | None = None,
    zlim: Tuple | None = None,
):
    """
    This is used when we want to plot an estimand that is assigned
    to all the points of the data.

    What this function does is that it takes all the estimated values
    and then performs a umap embedding (if needed) on data, then
    provides a heatmap where the intensity of a point represents the value
    of the estimand.

    To visualize LID values, for example, we can use this method.
    """

    # train a UMAP embedding on all the data
    assert data.shape[1] in [2, 3], "Only 2 and 3 dimensional data is covered"

    colors = [ColorTheme.RED_FIRST.value, ColorTheme.BLUE_FIRST.value, ColorTheme.GOLD.value]  #
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    estimates = np.clip(estimates, min_estimate, max_estimate)

    if data.shape[1] == 2:
        fig, ax = plt.subplots(figsize=figsize)
        scatter = sns.scatterplot(
            x=data[:, 0],
            y=data[:, 1],
            palette=cmap,
            legend=None,
            s=100,
            alpha=alpha,
            edgecolor="none",
            hue=estimates,
        )

    if data.shape[1] == 3:
        # Create a figure and a 3D axis
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        all_alphas = np.concatenate([alpha * np.ones(len(data)), np.zeros(2)])
        all_points = np.concatenate([data, np.array([[0, 0, -1000.5], [0, 0, +1000.5]])])
        all_estimates = np.concatenate([estimates, np.array([min_estimate, max_estimate])])
        ax.scatter(
            all_points[:, 0],
            all_points[:, 1],
            all_points[:, 2],
            c=all_estimates,
            cmap=cmap,
            alpha=all_alphas,
        )
        if box_ratios:
            # Adjust the aspect ratio to be equal
            ax.set_box_aspect(box_ratios)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if zlim:
        ax.set_zlim(*zlim)
    norm = plt.Normalize(vmin=min_estimate, vmax=max_estimate)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=ax,
        orientation="vertical",
        label=colorbar_label if colorbar_label is not None else "estimates",
    )
    if cbar_ticks:
        # Set colorbar ticks and tick labels
        tick_labels = [str(tick) for tick in cbar_ticks]
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(tick_labels)

    if custom_xticks:
        ax.set_xticks(custom_xticks)
    if custom_yticks:
        ax.set_yticks(custom_yticks)
    if custom_zticks:
        ax.set_zticks(custom_zticks)

    if remove_ticks_x_label:
        # Set tick parameters for the axes to white
        ax.tick_params(axis="x", colors="white")

    if remove_ticks_y_label:
        ax.tick_params(axis="y", colors="white")

    if remove_ticks_z_label and data.shape[1] == 3:
        ax.tick_params(axis="z", colors="white")

    ax.legend(loc=legend_loc, prop={"family": FONT_FAMILY, "size": legend_fontsize})
    if no_legend:
        ax.legend_.remove()

    if title is not None:
        ax.set_title(title, fontsize=fontsize, fontdict={"family": FONT_FAMILY})

    return ax


def visualize_umap_clusters(
    data: List[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray,
    labels: List[str] | str | None = None,
    title: str = "UMAP embeddings",
    alpha: List[float] | float = 0.1,
    colors: List[str] | None = None,
    return_img: bool = False,
    reducer: umap.UMAP | None = None,
    return_reducer: bool = False,
) -> Image:
    # Some checks
    if isinstance(data, (torch.Tensor, np.ndarray)):
        data = [data]
    if isinstance(labels, str):
        labels = [labels]
    assert len(data) == len(labels), "Data and labels should have the same length."
    if isinstance(alpha, float):
        alpha = [alpha] * len(data)
    assert len(alpha) == len(data), "Alpha and data should have the same length."

    # Turn everything into numpy arrays
    for i in range(len(data)):
        if isinstance(data[i], torch.Tensor):
            data[i] = data[i].cpu().numpy()

    # concatenate everything before passing to UMAP
    data_concatenated = np.concatenate(data, axis=0)
    # get the colors for visualizing the clusters
    colors = colors or ColorTheme.get_colors(len(data))

    # train a UMAP model and then visualize all the clusters
    try:
        if data_concatenated.shape[1] > 2:
            if not reducer:
                reducer = umap.UMAP()
                reducer.fit(data_concatenated)
            all_embeddings = reducer.transform(data_concatenated)
        else:
            all_embeddings = data_concatenated
        for i in range(len(data)):
            L = sum(len(d) for d in data[:i])
            R = sum(len(d) for d in data[: i + 1])
            embeddings = all_embeddings[L:R]
            plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=alpha, color=colors[i])
            plt.scatter([], [], color=colors[i], label=labels[i])
        plt.legend()
        plt.title(title)
    finally:
        img = None
        if not return_img:
            plt.show()
        else:
            fig = plt.gcf()
            fig.canvas.draw()
            np_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                fig.canvas.get_width_height()[::-1] + (3,)
            )
            # create a PIL image out of the np_array of size (H x W x 3)
            img = Image.fromarray(np_array)
        plt.close()
    return img if not return_reducer else (img, reducer)
