import functools

import pytest
import torch

from data.distributions import (
    AffineManifoldMixture,
    Lollipop,
    ManifoldMixture,
    MultiscaleGaussian,
    SquigglyManifoldMixture,
    SwissRoll,
    Torus,
)
from data.distributions.lid_base import LIDDistribution
from models.flows import RQNSF
from tests.utils import import_package_classes  # for coverage tests

rq_nsf_diffeo = functools.partial(
    RQNSF,
    n_transforms=10,
    n_hidden=32,
    n_blocks=3,
)

all_settings = [
    {
        "cls": Lollipop,
    },
    {
        "cls": SwissRoll,
    },
    {
        "cls": ManifoldMixture,
        "manifold_dims": [2, 3],
        "ambient_dim": 50,
        "seed": 42,
    },
    {
        "cls": ManifoldMixture,
        "manifold_dims": [2, 3, 10],
        "diffeomorphism_instantiator": [
            rq_nsf_diffeo,
            rq_nsf_diffeo,
            rq_nsf_diffeo,
        ],
        "ambient_dim": 50,
        "seed": 42,
    },
    {
        "cls": ManifoldMixture,
        "manifold_dims": [2, 3, 10],
        "diffeomorphism_instantiator": [
            rq_nsf_diffeo,
            None,
            None,
        ],
        "adjust_condition_number": True,
        "n_iter_calibration": 1,
        "ambient_dim": 50,
        "seed": 42,
    },
    {
        "cls": AffineManifoldMixture,
        "manifold_dims": [2, 3],
        "ambient_dim": 50,
        "affine_projection_type": "random-rotation",
        "seed": 43,
    },
    {
        "cls": SquigglyManifoldMixture,
        "manifold_dims": [2, 3],
        "ambient_dim": 50,
        "seed": 44,
        "kappa_control": 0.5,
    },
    {
        "cls": Torus,
    },
    {
        "cls": MultiscaleGaussian,
        "eigenvalues": [1.0, 0.1, 0.01],
    },
]


@pytest.fixture
def lid_distribution_classes():  # returns all the classes defined in the data.distributions that inherit LIDDistribution
    classes = import_package_classes("data.distributions")
    return [cls for cls in classes if issubclass(cls, LIDDistribution) and cls != LIDDistribution]


def test_coverage_lid_distribution(lid_distribution_classes):
    """
    This is a coverage test to ensure that everything in the package data.distributions
    that inherits from LIDDistribution is covered in the tests.
    """
    for cls in lid_distribution_classes:
        if issubclass(cls, LIDDistribution):
            assert cls in [setting["cls"] for setting in all_settings], f"{cls} not in all_settings"


# This should cover all the LIDDistribution classes in the codebase
@pytest.mark.parametrize(
    "lid_distributions",
    all_settings,
)
def test_lid_distributions(lid_distributions):
    x1 = None
    x2 = None
    x3 = None
    cls = lid_distributions.pop("cls")
    for i in range(3):
        distr = cls(**lid_distributions)

        assert isinstance(distr, LIDDistribution), f"{distr} is not an instance of LIDDistribution"

        dict = distr.sample(10, return_dict=True, seed=111)

        expected_keys = ["samples", "lid", "idx"]
        keys = list(dict.keys())
        assert all([key in keys for key in expected_keys]), f"{cls} dictionary unexpected: {keys}"
        assert all([key in expected_keys for key in keys]), f"{cls} dictionary unexpected: {keys}"

        # Test reproducibility using seed!
        x1_new = distr.sample(10, return_dict=False, seed=111)
        x2_new = dict["samples"]
        x3_new = distr.sample(10, return_dict=False, seed=111)
        if x1 is not None:
            assert torch.allclose(
                x1, x1_new
            ), f"reproducibility issue with: {cls}, x1 and x1_new not equal"
            assert torch.allclose(
                x2, x2_new
            ), f"reproducibility issue with: {cls}, x2 and x2_new not equal"
            assert torch.allclose(
                x3, x3_new
            ), f"reproducibility issue with: {cls}, x3 and x3_new not equal"
        x1, x2, x3 = x1_new, x2_new, x3_new
        assert torch.allclose(x1, x2), f"reproducibility issue with: {cls}, x1 and x2 not equal"
        assert torch.allclose(x1, x3), f"reproducibility issue with: {cls}, x1 and x3 not equal"
