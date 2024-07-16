import numpy as np


def mae(gt_lid: np.array, pred_lid: np.array) -> float:
    """
    Compute the mean absolute error between the ground truth LID and the estimated LID
    """
    mae = np.mean(np.abs(gt_lid - pred_lid))
    return float(mae)


def mse(gt_lid: np.array, pred_lid: np.array) -> float:
    """
    Compute the mean squared error between the ground truth LID and the estimated LID
    """
    mse = np.mean((gt_lid - pred_lid) ** 2)
    return float(mse)


def relative_bias(gt_lid: np.array, pred_lid: np.array, eps: float = 1e-3) -> float:
    """
    Compute the relative bias between the ground truth LID and the estimated LID
    """
    bias = np.mean(np.abs(gt_lid - pred_lid) / (gt_lid + eps))
    return float(bias if bias <= 1 else "inf")
