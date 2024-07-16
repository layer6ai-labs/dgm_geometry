import numpy as np
import pytest
import torch


def pytest_configure():
    pytest.ambient_dim = 6


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def gaussian_gt():
    ambient_dim = pytest.ambient_dim
    mu = torch.zeros(ambient_dim)
    cov_diag = torch.Tensor([8, 5, 9, 4, 1e-8, 1e-8])  # Intrinsic dim should be 4
    assert ambient_dim == mu.numel() == cov_diag.numel()

    return mu, cov_diag
