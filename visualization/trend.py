from typing import List, Optional
import functools

from cv2 import line
import matplotlib.pyplot as plt
import numpy as np
import PIL
import PIL.Image
import seaborn as sns
import pandas as pd

from .pretty import ColorTheme, FONT_FAMILY, hashlines, StyleDecorator, savable


def plot_trends(
    x_axis: np.array,  # shape (trend_length)
    y_axis: np.array,  # shape (num_trends, trend_length)
    cluster_idx: np.array,  # shape (num_trends)
    labels: List[str],  # mapping indices to the labels
    xlabel: str,
    ylabel: str,
    title: str,
    alpha: float = 0.1,
) -> PIL.Image:
    """
    This function takes in an x_axis and a set of y_axis trends
    for each trend, it visualizes the trend on the plot with the color
    associated with tits cluster_idx.
    """
    try:
        colors = ColorTheme.get_colors(count=len(labels))
        # group everything according to the cluster they belong to
        for trend, cluster_id in zip(y_axis, cluster_idx):
            plt.plot(x_axis, trend, alpha=alpha, color=colors[cluster_id])
        for cluster_id in np.unique(cluster_idx):
            msk = cluster_idx == cluster_id
            if not np.any(msk):
                continue
            plt.plot([], [], color=colors[cluster_id], label=labels[cluster_id])
            y_filtered = y_axis[msk]  # shape (num_trends_in_cluster, trend_length)
            avg_trend = y_filtered.mean(axis=0)
            plt.plot(x_axis, avg_trend, alpha=1.0, color=ColorTheme.PIRATE_BLACK.value)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
    finally:
        img = None
        fig = plt.gcf()
        fig.canvas.draw()
        np_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )
        # create a PIL image out of the np_array of size (H x W x 3)
        img = PIL.Image.fromarray(np_array)
        plt.close()
    return img


def _plot_trends(
    t_values: np.array,
    mean_values: List[np.array],
    labels: List[str],
    colors: List,
    std_values: Optional[List[np.array]] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    figsize: Optional[tuple] = (10, 6),
    with_std: bool = False,
    with_avg_in_legend: bool = True,
    vertical_lines: Optional[List[float]] = None,
    vertical_line_thickness: Optional[float] = None,
    vertical_lines_color: Optional[List] = None,
    horizontal_lines: Optional[List[float]] = None,
    horizontal_lines_thickness: Optional[float] = None,
    horizontal_lines_color: Optional[List] = None,
    smoothing_window: int = 1,
    fontsize: Optional[int] = None,
    tick_fontsize: Optional[int] = None,
    custom_xticks: Optional[int] = None,
    custom_yticks: Optional[int] = None,
    fake_yticks: Optional[int] = None,
    no_legend: bool = False,
    legend_fontsize: Optional[int] = None,
    legend_loc: Optional[str] = None,
    linewidth: float | None = None,
    alpha: float | None = None,
    title: str | None = None,
):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    idx = 0

    smoothing_window_means = []
    smoothing_window_stds = []

    if std_values is None:
        std_values = [None for _ in mean_values]

    for means, stds, color, lbl in zip(mean_values, std_values, colors, labels):
        smoothing_window_means.append(means)
        if stds is not None:
            smoothing_window_stds.append(stds)
        if len(smoothing_window_stds) > smoothing_window:
            smoothing_window_means.pop(0)
            if stds is not None:
                smoothing_window_stds.pop(0)

        smooth_mean = sum(smoothing_window_means) / len(smoothing_window_means)
        if stds is not None:
            smooth_std = sum(smoothing_window_stds) / len(smoothing_window_stds)

        # Create a lineplot using Seaborn
        my_lbl = f"Avg. {lbl}" if with_avg_in_legend else lbl
        sns.lineplot(
            x=t_values,
            y=smooth_mean,
            color=color,
            ax=ax,
            label=my_lbl,
            linewidth=linewidth,
            alpha=alpha,
        )

        # Use fill_between to add the transparent area representing std
        if stds is not None:
            if with_std:
                ax.fill_between(
                    t_values,
                    smooth_mean - smooth_std,
                    smooth_mean + smooth_std,
                    alpha=0.3,
                    label=f"std {lbl}",
                    color=color,
                    hatch=hashlines[idx % len(hashlines)],
                )
            else:
                ax.fill_between(
                    t_values,
                    smooth_mean - smooth_std,
                    smooth_mean + smooth_std,
                    alpha=0.3,
                    color=color,
                    hatch=hashlines[idx % len(hashlines)],
                )
        idx += 1

    if vertical_lines is not None:
        if vertical_lines_color is None:
            vertical_lines_color = [ColorTheme.PIRATE_GOLD.value for _ in vertical_lines]
        for vert, color in zip(vertical_lines, vertical_lines_color):
            # Add a vertical dotted line at x=0.5 with color green
            ax.axvline(
                x=vert,
                color=color,
                linestyle="--",
                linewidth=vertical_line_thickness,
            )

    if horizontal_lines is not None:
        for hor, col in zip(horizontal_lines, horizontal_lines_color):
            ax.axhline(y=hor, color=col, linestyle=":", linewidth=horizontal_lines_thickness)

    # Set labels, title, and legend
    if x_label:
        ax.set_xlabel(x_label, fontsize=fontsize, fontdict={"family": FONT_FAMILY})
    if y_label:
        ax.set_ylabel(y_label, fontsize=fontsize, fontdict={"family": FONT_FAMILY})

    # Adjusting tick font size
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    if custom_xticks:
        ax.set_xticks(custom_xticks)
    if custom_yticks:
        ax.set_yticks(custom_yticks)
        if fake_yticks:
            ax.set_yticklabels([f"{i}" for i in fake_yticks])

    ax.legend(loc=legend_loc, prop={"family": FONT_FAMILY, "size": legend_fontsize})
    if no_legend:
        ax.legend_.remove()

    if title is not None:
        ax.set_title(title, fontsize=fontsize, fontdict={"family": FONT_FAMILY})

    return ax


# def _plot_trends(
#     t_values: np.array,
#     mean_values: List[np.array],
#     labels: List[str],
#     colors: List,
#     styles: List[str] | None = None,
#     std_values: Optional[List[np.array]] = None,
#     x_label: Optional[str] = None,
#     y_label: Optional[str] = None,
#     figsize: Optional[tuple] = (10, 6),
#     with_std: bool = False,
#     with_avg_in_legend: bool = True,
#     vertical_lines: Optional[List[float]] = None,
#     vertical_line_thickness: Optional[float] = None,
#     vertical_lines_color: Optional[List] = None,
#     horizontal_lines: Optional[List[float]] = None,
#     horizontal_lines_thickness: Optional[float] = None,
#     horizontal_lines_color: Optional[List] = None,
#     smoothing_window: int = 1,
#     fontsize: Optional[int] = None,
#     tick_fontsize: Optional[int] = None,
#     custom_xticks: Optional[int] = None,
#     custom_yticks: Optional[int] = None,
#     fake_yticks: Optional[int] = None,
#     no_legend: bool = False,
#     legend_fontsize: Optional[int] = None,
#     legend_loc: Optional[str] = None,
#     title: str | None = None,
#     linewidth: Optional[int] = None,
#     alpha: Optional[float] = None,
# ):
#     # Create a figure and axis
#     fig, ax = plt.subplots(figsize=figsize)
#     idx = 0

#     smoothing_window_means = []
#     smoothing_window_stds = []

#     if std_values is None:
#         std_values = [None for _ in mean_values]

#     if styles is None:
#         styles = ["-" for _ in mean_values]

#     for means, stds, color, lbl, style in zip(mean_values, std_values, colors, labels, styles):
#         smoothing_window_means.append(means)
#         if stds is not None:
#             smoothing_window_stds.append(stds)
#         if len(smoothing_window_stds) > smoothing_window:
#             smoothing_window_means.pop(0)
#             if stds is not None:
#                 smoothing_window_stds.pop(0)

#         smooth_mean = sum(smoothing_window_means) / len(smoothing_window_means)
#         if stds is not None:
#             smooth_std = sum(smoothing_window_stds) / len(smoothing_window_stds)

#         # Create a lineplot using Seaborn
#         lineplot_lbl = f"Avg. {lbl}" if with_avg_in_legend else lbl
#         # create a dataframe out of t_values, smooth_mean
#         df = pd.DataFrame({"t": t_values, "mean": smooth_mean, "styles": [style] * len(t_values)})
#         sns.lineplot(
#             data=df,
#             x="t",
#             y="mean",
#             color=color,
#             ax=ax,
#             # label=lineplot_lbl if stds is not None else lbl,
#             linewidth=linewidth,
#             alpha=alpha,
#             style="styles",
#         )

#         # Use fill_between to add the transparent area representing std
#         if stds is not None:
#             if with_std:
#                 ax.fill_between(
#                     t_values,
#                     smooth_mean - smooth_std,
#                     smooth_mean + smooth_std,
#                     alpha=0.3,
#                     label=f"std {lbl}",
#                     color=color,
#                     hatch=hashlines[idx % len(hashlines)],
#                 )
#             else:
#                 ax.fill_between(
#                     t_values,
#                     smooth_mean - smooth_std,
#                     smooth_mean + smooth_std,
#                     alpha=0.3,
#                     color=color,
#                     hatch=hashlines[idx % len(hashlines)],
#                 )
#         idx += 1

#     if vertical_lines is not None:
#         if vertical_lines_color is None:
#             vertical_lines_color = [ColorTheme.PIRATE_GOLD.value for _ in vertical_lines]
#         for vert, color in zip(vertical_lines, vertical_lines_color):
#             # Add a vertical dotted line at x=0.5 with color green
#             ax.axvline(
#                 x=vert,
#                 color=color,
#                 linestyle="--",
#                 linewidth=vertical_line_thickness,
#             )

#     if horizontal_lines is not None:
#         for hor, col in zip(horizontal_lines, horizontal_lines_color):
#             ax.axhline(
#                 y=hor, color=col, linestyle=":", linewidth=horizontal_lines_thickness
#             )

#     # Set labels, title, and legend
#     if x_label:
#         ax.set_xlabel(x_label, fontsize=fontsize, fontdict={"family": FONT_FAMILY})
#     if y_label:
#         ax.set_ylabel(y_label, fontsize=fontsize, fontdict={"family": FONT_FAMILY})

#     # Adjusting tick font size
#     ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

#     if custom_xticks:
#         ax.set_xticks(custom_xticks)
#     if custom_yticks:
#         ax.set_yticks(custom_yticks)
#         if fake_yticks:
#             ax.set_yticklabels([f"{i}" for i in fake_yticks])

#     ax.legend(loc=legend_loc, prop={"family": FONT_FAMILY, "size": legend_fontsize})
#     if no_legend:
#         ax.legend_.remove()

#     if title is not None:
#         ax.set_title(title, fontsize=fontsize, fontdict={"family": FONT_FAMILY})

#     return ax


@savable
@StyleDecorator(font_scale=2, style="white", line_style="--")
@functools.wraps(_plot_trends)
def plot_trends_no_precision(
    *args,
    **kwargs,
):
    return _plot_trends(*args, **kwargs)


@savable
@StyleDecorator(font_scale=2, style="whitegrid", line_style="--")
@functools.wraps(_plot_trends)
def plot_trends_with_precision(
    *args,
    **kwargs,
):
    return _plot_trends(*args, **kwargs)


@savable
@StyleDecorator(font_scale=2, style="whitegrid", line_style="--")
def grouping_trends(
    t_values: np.array,
    trends: List[np.array],
    hues: List[str],
    styles: List[str],
    hue_name: str,
    style_name: str,
    # colors: List,
    std_values: Optional[List[np.array]] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    figsize: Optional[tuple] = (10, 6),
    smoothing_window: int = 1,
    fontsize: Optional[int] = None,
    tick_fontsize: Optional[int] = None,
    custom_xticks: Optional[int] = None,
    custom_yticks: Optional[int] = None,
    fake_yticks: Optional[int] = None,
    no_legend: bool = False,
    legend_fontsize: Optional[int] = None,
    legend_loc: Optional[str] = None,
    title: str | None = None,
    linewidth: Optional[int] = None,
    alpha: Optional[float] = None,
):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    smoothing_window_means = []
    smoothing_window_stds = []

    # create a dataframe out of t_values, smooth_mean
    all_t_values = []
    all_hue_values = []
    all_style_values = []
    all_trend_values = []
    for means, hue, style in zip(trends, hues, styles):
        smoothing_window_means.append(means)
        if len(smoothing_window_stds) > smoothing_window:
            smoothing_window_means.pop(0)
        smooth_mean = sum(smoothing_window_means) / len(smoothing_window_means)
        all_t_values.extend(t_values)
        all_hue_values.extend([hue] * len(t_values))
        all_style_values.extend([style] * len(t_values))
        all_trend_values.extend(smooth_mean)
        # df["hue"] = [hue] * len(t_values)
        # df["style"] = [style] * len(t_values)
        # df[lbl] = smooth_mean

    sns.lineplot(
        data=pd.DataFrame(
            {
                "t": all_t_values,
                "mean": all_trend_values,
                hue_name: all_hue_values,
                style_name: all_style_values,
            }
        ),
        x="t",
        y="mean",
        hue=hue_name,
        style=style_name,
        # color=color,
        ax=ax,
        # label=lineplot_lbl if stds is not None else lbl,
        linewidth=linewidth,
        alpha=alpha,
        palette=ColorTheme.get_colors(count=len(set(hues))),
    )

    # Set labels, title, and legend
    if x_label:
        ax.set_xlabel(x_label, fontsize=fontsize, fontdict={"family": FONT_FAMILY})
    if y_label:
        ax.set_ylabel(y_label, fontsize=fontsize, fontdict={"family": FONT_FAMILY})

    # Adjusting tick font size
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    if custom_xticks:
        ax.set_xticks(custom_xticks)
    if custom_yticks:
        ax.set_yticks(custom_yticks)
        if fake_yticks:
            ax.set_yticklabels([f"{i}" for i in fake_yticks])

    ax.legend(loc=legend_loc, prop={"family": FONT_FAMILY, "size": legend_fontsize})
    if no_legend:
        ax.legend_.remove()

    if title is not None:
        ax.set_title(title, fontsize=fontsize, fontdict={"family": FONT_FAMILY})

    return ax
