import pytest
import torch

from data.distributions import VonMisesEuclidean


@pytest.fixture
def vonmises():
    return VonMisesEuclidean()


def test_vonmises_1d(vonmises):
    data = vonmises.sample((3,))
    assert data.shape == (3, 2)
    radii = data[..., 0] ** 2 + data[..., 1] ** 2
    assert torch.isclose(radii, torch.ones_like(radii)).all()  # Points should lie on circle


def test_vonmises_2d(vonmises):
    data = vonmises.sample((3, 3))
    assert data.shape == (3, 3, 2)
    radii = data[..., 0] ** 2 + data[..., 1] ** 2
    assert torch.isclose(radii, torch.ones_like(radii)).all()  # Points should lie on circle
