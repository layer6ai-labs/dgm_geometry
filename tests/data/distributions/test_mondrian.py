import pytest

from data.distributions import Mondrian


@pytest.fixture
def mondrian():
    return Mondrian()


def test_mondrian_2d(mondrian):
    data = mondrian.sample((3, 3))
    assert data.shape == (3, 3, 3, 32, 32)
