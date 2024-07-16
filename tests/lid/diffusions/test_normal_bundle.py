import pytest
import torch

from lid.diffusions import NormalBundleEstimator
from tests.models.diffusions.sdes.test_sdes import ve_sde


def test_stanczuk_estimator(ve_sde):
    # should specify ambient_dim from now on!
    lid_estimator = NormalBundleEstimator(model=ve_sde, ambient_dim=6, device=torch.device("cpu"))

    x1 = torch.zeros((1, 6))
    x2 = torch.zeros((2, 6))
    x3 = torch.zeros((6, 6))
    # Test with some different shapes
    assert torch.equal(lid_estimator.estimate_lid(x1), torch.Tensor([4]))
    assert torch.equal(lid_estimator.estimate_lid(x2), torch.Tensor([4, 4]))
    assert torch.equal(lid_estimator.estimate_lid(x3), torch.full((6,), 4))
