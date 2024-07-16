import numpy as np
import typing as th
from tqdm import tqdm


def get_roc_graph(
    pos_x: th.Optional[np.ndarray] = None,
    pos_y: th.Optional[np.ndarray] = None,
    neg_x: th.Optional[np.ndarray] = None,
    neg_y: th.Optional[np.ndarray] = None,
    compute_limit: th.Optional[int] = None,
    verbose: int = 1,
):
    if pos_x is None or neg_x is None:
        raise Exception("pos_x and neg_x should be defined")

    if pos_y is None or neg_y is None:
        pos_y = np.ones_like(pos_x)
        neg_y = np.ones_like(neg_x)

    N = len(pos_x)

    all_x = np.concatenate([pos_x, neg_x, np.array([np.min(pos_x) + 1e-6, np.max(neg_x) + 1e-6])])
    all_y = np.concatenate([pos_y, neg_y, np.array([np.min(pos_y) + 1e-6, np.max(neg_y) + 1e-6])])
    all_x = np.unique(all_x)
    all_y = np.unique(all_y)
    all_x, all_y = np.meshgrid(all_x, all_y)
    all_x = all_x.flatten()
    all_y = all_y.flatten()

    if compute_limit:
        compute_limit = min(compute_limit, len(all_x))
        msk = np.array([True] * compute_limit + [False] * (len(all_x) - compute_limit))
        np.random.shuffle(msk)
        all_x = all_x[msk]
        all_y = all_y[msk]

    points = []

    rng = zip(all_x, all_y)
    if verbose > 0:
        rng = tqdm(rng, total=len(all_x))

    for x, y in rng:
        # the classifier is >x and >y
        tp = np.sum((pos_x >= x) & (pos_y >= y)) / len(pos_x)
        fp = np.sum((neg_x >= x) & (neg_y >= y)) / len(neg_x)
        points.append((fp, tp))
    sorted_points = sorted(points, key=lambda x: "{:.10f}_{:.10f}".format(x[0], x[1]))
    x, y = map(np.array, zip(*sorted_points))

    return x, y


def get_pareto_frontier(graph_x, graph_y):

    ret_x = [0.0]
    ret_y = [0.0]
    for x, y in zip(graph_x, graph_y):
        if x >= ret_x[-1] and y >= ret_y[-1]:
            ret_x.append(x)
            ret_y.append(y)

    return np.array(ret_x + [1.0]), np.array(ret_y + [1.0])


def get_auc(curve_x, curve_y):
    """
    Given a curve, return the area under the curve
    """
    auc = 0.0

    for i in range(1, len(curve_x)):

        auc += (curve_x[i] - curve_x[i - 1]) * (curve_y[i] + curve_y[i - 1]) / 2.0
    return auc
