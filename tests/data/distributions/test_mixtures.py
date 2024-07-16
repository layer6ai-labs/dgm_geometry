import functools

import pytest
import torch
from skdim.id import MLE, lPCA

from data.distributions import AffineManifoldMixture, ManifoldMixture, SquigglyManifoldMixture
from models.flows.diffeomorphisms import RQNSF


@pytest.mark.parametrize(
    "freq",
    [0.1, 1.0, 20.0],
)
def test_squiggly_manifolds(freq):
    """
    Create three different squiggly manifolds and compute their lPCA lid estimates.
    """
    manifold_dims = [1, 2, 3]
    distribution = SquigglyManifoldMixture(
        manifold_dims=[1, 2, 3],
        ambient_dim=3,
        distance_between_modes=10,
        sample_distr="uniform",
        seed=53,
        frequency=freq,
        n_transforms=5,
    )
    ret = distribution.sample((1000,), chunk_size=1000, return_dict=True)
    data = ret["samples"]
    idx = ret["idx"]

    for i in range(len(manifold_dims)):
        id_lpca = lPCA().fit(data[idx == i].numpy()).dimension_
        assert manifold_dims[i] == round(
            id_lpca
        ), f"lPCA ID of mode {i} is {id_lpca} but should be {manifold_dims[i]}"
        id_mle = MLE().fit(data[idx == i].numpy()).dimension_
        assert manifold_dims[i] == round(
            id_lpca
        ), f"MLE ID of mode {i} is {id_mle} but should be {manifold_dims[i]}"


def test_affine_mixtures():
    """
    Create a mixture of 3 Gaussians using the mixture of affine manifolds code
    """
    distribution = AffineManifoldMixture(
        manifold_dims=[10, 100, 150],
        ambient_dim=800,
        affine_projection_type="random",
    )
    distribution.sample(10)


def test_manifold_mixtures_reproducibility():
    """
    This is a very strong test to check that even if we train the condition number,
    does it affect reproducibility or not.
    """
    rq_nsf_diffeo = functools.partial(
        RQNSF,
        n_transforms=10,
        n_hidden=32,
        n_blocks=3,
    )
    distribution1 = ManifoldMixture(
        manifold_dims=[10, 20],
        ambient_dim=40,
        diffeomorphism_instantiator=[rq_nsf_diffeo, rq_nsf_diffeo],
        affine_projection_type="random-rotation",
        distance_between_modes=6,
        sample_distr="uniform",
        seed=666,
        adjust_condition_number=True,
        n_calibration=10,
        n_iter_calibration=1,
    )
    data1 = distribution1.sample((128,), chunk_size=128, seed=100)

    distribution2 = ManifoldMixture(
        manifold_dims=[10, 20],
        ambient_dim=40,
        diffeomorphism_instantiator=[rq_nsf_diffeo, rq_nsf_diffeo],
        affine_projection_type="random-rotation",
        distance_between_modes=6,
        sample_distr="uniform",
        seed=666,
        adjust_condition_number=True,
        n_calibration=10,
        n_iter_calibration=1,
    )
    data2 = distribution2.sample((128,), chunk_size=128, seed=100)
    assert torch.allclose(data1, data2), "Data should be the same!"


@pytest.mark.parametrize(
    "projection_type",
    ["random", "random-rotation", "zero-pad", "repeat"],
)
@pytest.mark.parametrize(
    "sample_distr",
    ["uniform", "normal", "laplace"],
)
def test_manifold_mixtures(projection_type, sample_distr):
    """
    Create a complex diffeomorphism and create a mixture of Riemanian manifolds using that
    """
    rq_nsf_diffeo = functools.partial(
        RQNSF,
        n_transforms=10,
        n_hidden=32,
        n_blocks=3,
    )
    distribution = ManifoldMixture(
        manifold_dims=[10, 20],
        ambient_dim=40,
        diffeomorphism_instantiator=[rq_nsf_diffeo, rq_nsf_diffeo],
        affine_projection_type=projection_type,
        distance_between_modes=6,
        sample_distr=sample_distr,
        seed=666,
        adjust_condition_number=False,
    )
    distribution.sample((128,), chunk_size=128)
