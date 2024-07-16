"""
Different tests for testing the JVB-based trace calculation of a Jacobian matrix.
"""

from typing import Callable, List, Tuple

import pytest
import torch

from models.diffusions.sdes.utils import compute_trace_of_jacobian


@pytest.fixture
def functions_and_jacobians() -> List[
    Tuple[
        Callable[[torch.Tensor], torch.Tensor],
        Callable[[torch.Tensor], torch.Tensor],
    ]
]:
    """returns a list of functions and their Jacobians"""
    fn_x = lambda x: x
    fn_x2 = lambda x: x**2

    def fn_complex(x: torch.Tensor):
        # replace x with a rotated version of x
        return x * (torch.roll(x, 1, dims=-1) ** 2) * (torch.roll(x, 2, dims=-1) ** 3)

    jacobian_fn_x = lambda x: torch.stack([torch.diag(torch.ones_like(x_batch)) for x_batch in x])
    jacobian_fn_x2 = lambda x: torch.stack([torch.diag(2 * x_batch) for x_batch in x])

    def jacobian_fn_complex(x: torch.Tensor):
        # y[i] = x[i] * x[i-1]**2 * x[i-2]**3
        jac = torch.zeros((x.shape[0], x.shape[1], x.shape[1]), dtype=x.dtype, device=x.device)
        for b, x_batch in enumerate(x):
            for i in range(len(x_batch)):
                i_1 = (i - 1 + len(x_batch)) % len(x_batch)
                i_2 = (i - 2 + len(x_batch)) % len(x_batch)
                jac[b, i, i] = x_batch[i_1] ** 2 * x_batch[i_2] ** 3
                jac[b, i, i_1] = 2 * x_batch[i] * x_batch[i_1] * x_batch[i_2] ** 3
                jac[b, i, i_2] = 3 * x_batch[i] * x_batch[i_1] ** 2 * x_batch[i_2] ** 2
        return jac

    return [
        (fn_x, jacobian_fn_x),
        (fn_x2, jacobian_fn_x2),
        (fn_complex, jacobian_fn_complex),
    ]


# Set different methods for trace estimation with different absolute error tolerances
@pytest.mark.parametrize(
    "setting",
    [
        ({"method": "hutchinson_rademacher", "hutchinson_sample_count": 10000}, 1e-1),
        ({"method": "hutchinson_gaussian", "hutchinson_sample_count": 10000}, 1e-1),
        ({"method": "deterministic"}, 1e-6),
        ({"method": None}, 1e-6),
    ],
)
def test_jvp_based_trace_of_jacobian(functions_and_jacobians, setting):
    trace_hyperparameters, all_close_epsilon = setting
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [10.0, 100.0, -1.0],
        ]
    )
    for fn, jacobian_fn in functions_and_jacobians:
        jacobian = jacobian_fn(x)
        expected_traces = torch.stack([torch.trace(jac) for jac in jacobian])
        computed_traces = compute_trace_of_jacobian(fn=fn, x=x, **trace_hyperparameters)
        relative_errors = torch.abs(computed_traces - expected_traces) / expected_traces
        assert torch.allclose(
            relative_errors, torch.zeros_like(expected_traces), atol=all_close_epsilon
        ), f"Relative trace errors:  {relative_errors}"
