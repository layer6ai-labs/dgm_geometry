import numpy as np
import pytest

from lid.evaluation import concordance_index, mae, mse, relative_bias


@pytest.mark.parametrize("seed", [10, 20, 30])
def test_evaluation_metrics(seed):

    np.random.seed(seed)
    ambient_dim = 100
    N = 1000
    pred_lid = np.random.rand(N) * ambient_dim
    gt_lid = np.round(np.random.rand(N) * ambient_dim)

    expected_mae = np.mean(np.abs(gt_lid - pred_lid))
    assert mae(gt_lid, pred_lid) == expected_mae
    expected_mse = np.mean((gt_lid - pred_lid) ** 2)
    assert mse(gt_lid, pred_lid) == expected_mse
    expected_relative_bias = np.mean(np.abs(gt_lid - pred_lid) / (gt_lid + 1e-3))
    expected_relative_bias = float("inf") if expected_relative_bias > 1 else expected_relative_bias
    assert relative_bias(gt_lid, pred_lid, eps=1e-3) == expected_relative_bias
    N //= 10
    ambient_dim //= 10
    pred_lid = np.random.rand(N) * ambient_dim
    gt_lid = np.round(np.random.rand(N) * ambient_dim)

    # create a comparison metric for gt_lid where [i, j]th element is 1 if gt_lid[i] > gt_lid[j] and 0 otherwise
    err = 0
    for i in range(N):
        for j in range(N):
            if gt_lid[i] > gt_lid[j] and pred_lid[i] < pred_lid[j]:
                err += 1
    expected_concordance_index = 1 - err / (N * (N - 1) / 2)
    assert (
        abs(concordance_index(gt_lid, pred_lid) - expected_concordance_index) < 1e-4
    ), f"got {concordance_index(integer_gt_lid, integer_pred_lid, with_equal=True)} expected {expected_concordance_index}"

    integer_gt_lid = np.random.randint(0, ambient_dim, N)
    integer_pred_lid = np.random.randint(0, ambient_dim, N)
    correct_pairs = 0
    for i in range(N):
        for j in range(i):
            gt_state = 0
            if integer_gt_lid[i] > integer_gt_lid[j]:
                gt_state = 1
            if integer_gt_lid[i] < integer_gt_lid[j]:
                gt_state = 2

            pred_state = 0
            if integer_pred_lid[i] > integer_pred_lid[j]:
                pred_state = 1
            if integer_pred_lid[i] < integer_pred_lid[j]:
                pred_state = 2
            correct_pairs += gt_state == pred_state
    expected_concordance_index = float(correct_pairs) / (N * (N - 1) / 2)
    assert (
        abs(
            concordance_index(integer_gt_lid, integer_pred_lid, with_equal=True)
            - expected_concordance_index
        )
        < 1e-4,
        f"got {concordance_index(integer_gt_lid, integer_pred_lid, with_equal=True)} expected {expected_concordance_index}",
    )


def test_concordance_index():
    # TODO: fail if it takes more than 5 seconds
    np.random.seed(42)
    ambient_dim = 100
    N = 500000
    pred_lid = np.random.rand(N) * ambient_dim
    # turn pred_lid into float32
    gt_lid = np.round(pred_lid)
    assert (
        concordance_index(gt_lid, pred_lid) == 1.0
    ), "Concordance index should be 1.0 for perfectly correlated data"
