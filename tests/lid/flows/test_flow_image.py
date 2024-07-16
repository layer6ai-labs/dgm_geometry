import functools

import pytest
import torch
from nflows.distributions import StandardNormal
from nflows.flows.base import Flow

from lid.flows import FastFlowLIDL, JacobianThresholdEstimator
from models.flows import RQNSF
from models.flows.diffeomorphisms import MultiscaleImageFlow


@pytest.fixture
def flow_rqnsf():
    torch.manual_seed(0)
    transform = MultiscaleImageFlow(
        H=4,
        W=4,
        C=1,
        squeezing_factors=[2, 2],
        n_blocks=2,
        coupling_partial=functools.partial(
            RQNSF,
            n_hidden=3,
            n_blocks=2,
            activation=torch.nn.functional.relu,
            dropout_probability=0.2,
            use_batch_norm=False,
            tails="linear",
            num_bins=2,
            tail_bound=1.0,
            data_type="image",
        ),
    )
    return Flow(
        transform=transform,
        distribution=StandardNormal(shape=[16]),
    ).eval()


@pytest.fixture
def image_like():
    torch.manual_seed(0)
    return torch.randn((10, 1, 4, 4))


def test_flow_lid_image(image_like, flow_rqnsf):
    lid_estimator1 = FastFlowLIDL(model=flow_rqnsf, ambient_dim=16)
    lid1 = lid_estimator1.estimate_lid(image_like, delta=-5)
    assert lid1.shape == (10,)
    lid_estimator2 = JacobianThresholdEstimator(model=flow_rqnsf, ambient_dim=16)
    lid2 = lid_estimator2.estimate_lid(image_like)
    assert lid2.shape == (10,)
    lid3 = lid_estimator2.estimate_lid(image_like, singular_value_threshold=-5)
    assert lid3.shape == (10,)
