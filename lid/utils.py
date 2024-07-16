from typing import Iterable, Tuple

import numpy as np
import torch
from kneed import KneeLocator


def fast_regression(
    all_ys: torch.Tensor,  # [data_size, trajectory_size]
    xs: torch.Tensor,  # [trajectory_size]
) -> torch.Tensor:  # [data_size]
    """Perform a rapid 1D linear regression that can utilize GPU as well"""
    column_with_nan = torch.isnan(all_ys).any(dim=0)
    xs = xs[~column_with_nan]
    all_ys = all_ys[:, ~column_with_nan]

    # return a tensor of slopes of size [data_size]
    mean_x = xs.mean()
    x_diff = xs - mean_x
    denomenator = torch.sum(x_diff**2)
    y_diff = all_ys - all_ys.mean(dim=1, keepdim=True)  # [data_size, trajectory_size]

    return torch.sum(y_diff * x_diff, dim=-1) / denomenator
