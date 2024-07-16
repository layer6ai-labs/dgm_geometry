"""
Testing the interface and abstraction of the LID estimator methods.
"""

import pytest
import torch
from nflows.distributions import StandardNormal
from nflows.flows.base import Flow

from data.distributions import Lollipop
from lid.base import ModelBasedLIDEstimator
from lid.diffusions import NormalBundleEstimator
from lid.flows import FastFlowLIDL, JacobianThresholdEstimator
from lid.flows.jacobian_flow import JacobianFlowLIDEstimator
from models.diffusions.networks import SimpleDiffusionMLP
from models.diffusions.sdes import VpSde
from models.flows import AffineFlow
from tests.utils import import_package_classes  # for coverage tests


@pytest.fixture
def lollipop_dset() -> torch.Tensor:

    torch.manual_seed(0)
    data = Lollipop().sample((10000, 2))
    return torch.utils.data.TensorDataset(data)


@pytest.fixture
def vp_sde() -> VpSde:
    torch.manual_seed(0)
    network = SimpleDiffusionMLP(2, hidden_sizes=(1024, 1024))
    return VpSde(network)  # This class contains the logic for the actual SDE


@pytest.fixture
def vp_sde3() -> VpSde:
    torch.manual_seed(0)
    network = SimpleDiffusionMLP(3, hidden_sizes=(1024, 1024))
    return VpSde(network)  # This class contains the logic for the actual SDE


@pytest.fixture
def flow_affine():
    torch.manual_seed(0)
    transform = AffineFlow(
        dim=2,
        n_transforms=3,
        n_hidden=3,
        n_blocks=2,
        data_type="tabular",
        include_actnorm=True,
        flip=False,
    )
    model = Flow(
        transform=transform,
        distribution=StandardNormal(shape=[2]),
    )
    return model


all_diffusion_based_estimators = [
    NormalBundleEstimator,
]

all_flow_based_estimators = [
    JacobianThresholdEstimator,
    FastFlowLIDL,
]


@pytest.fixture
def get_all_model_based_estimators():
    classes = import_package_classes("lid")
    return [
        cls
        for cls in classes
        if cls != ModelBasedLIDEstimator
        and issubclass(cls, ModelBasedLIDEstimator)
        and cls != JacobianFlowLIDEstimator
    ]


def test_coverage_model_based_lid(get_all_model_based_estimators):
    # check all in get_all_model_based_estimators are in all_model_based_estimators
    assert set(get_all_model_based_estimators) == set(all_diffusion_based_estimators).union(
        all_flow_based_estimators
    )


# The lid_class should cover all the model_based lid estimators in the codebase
@pytest.mark.parametrize("lid_class", all_diffusion_based_estimators)
def test_model_based_interface_diffusion(lid_class, lollipop_dset, vp_sde, vp_sde3):

    lid_estimator = lid_class(model=vp_sde, ambient_dim=2)
    with pytest.raises(AssertionError):
        lid_estimator.fit()

    with pytest.raises(AssertionError):
        lid_estimator.estimate_lid(torch.zeros((1, 3)))
    lid_estimator.estimate_lid(lollipop_dset[:10][0])

    lid_estimator = lid_class(model=vp_sde3)
    lid_estimator.estimate_lid(torch.zeros((1, 3)))
    with pytest.raises(AssertionError):
        lid_estimator.estimate_lid(lollipop_dset[:10][0])


@pytest.mark.parametrize("lid_class", all_flow_based_estimators)
def test_flow_based_interface_flow(lid_class, lollipop_dset, flow_affine):

    lid_estimator = lid_class(model=flow_affine, ambient_dim=2)
    with pytest.raises(AssertionError):
        lid_estimator.fit()

    # NOTE: in reality, one should not pass in both delta and singular_value_threshold at the same time
    with pytest.raises(AssertionError):
        lid_estimator.estimate_lid(torch.zeros((1, 3)), delta=-5, singular_value_threshold=-5)
    lid_estimator.estimate_lid(lollipop_dset[:10][0], delta=-5, singular_value_threshold=-5)

    if isinstance(lid_estimator, FastFlowLIDL):
        with pytest.raises(AssertionError):  # call without delta
            lid_estimator.estimate_lid(lollipop_dset[:10][0])
    elif isinstance(lid_estimator, JacobianThresholdEstimator):
        lid_estimator.estimate_lid(lollipop_dset[:10][0])
    else:
        raise ValueError("Unknown LID estimator")
