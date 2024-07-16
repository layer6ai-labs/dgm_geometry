"""
Test the log_prob function using a VpSde module where the score network is not trained, but set analytically.
"""

from typing import List, Tuple

import pytest
import torch

from models.diffusions.sdes import VeSde, VpSde
from models.diffusions.sdes.utils import VpSdeGaussianAnalytical


@pytest.fixture
def get_score_network() -> Tuple[List[torch.Tensor], List[float], List[VpSdeGaussianAnalytical]]:
    """returns a list of perfectly correct score networks"""
    torch.manual_seed(42)

    means = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([10.0, 0.1]),
        torch.randn(3),
    ]
    covs = [
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.randn((2, 2)),
        torch.randn((3, 3)),
    ]
    vpsdes = []
    times = [0.1, 0.05, 0.3]
    samples = []
    for mean, cov in zip(means, covs):
        beta_min = 0.1
        beta_max = 20.0
        t_max = 1.0
        vpsdes.append(
            VpSde(
                score_net=VpSdeGaussianAnalytical(
                    posterior_mean=mean,
                    posterior_cov=cov,
                    beta_max=beta_max,
                    beta_min=beta_min,
                    t_max=t_max,
                ),
                beta_max=beta_max,
                beta_min=beta_min,
                t_max=t_max,
            )
        )
        # take random samples
        samples.append(torch.randn(100, mean.numel()))
    return samples, times, vpsdes


@pytest.fixture
def ve_sde(gaussian_gt):
    mu, cov_diag = gaussian_gt
    ve_sde = VeSde(None)  # Will construct the score net analytically and add it after

    def score_net(x, t):
        # Theoretically perfect score network for VE SDE applied to N(mu, cov_diag)
        assert x.ndim == 2, "Input must have batch dimension."

        sigmas = ve_sde.sigma(t)
        convolved_vars = cov_diag + sigmas[..., None] ** 2
        return -(1 / convolved_vars) * (x - mu[None, :]) * sigmas[:, None]

    ve_sde.score_net = score_net
    return ve_sde


@pytest.fixture
def vp_sde(gaussian_gt):
    mu, cov_diag = gaussian_gt
    vp_sde = VpSde(None)  # Will construct the score net analytically and add it after

    def score_net(x, t):
        # Theoretically perfect score network for VP SDE applied to N(mu, cov_diag)
        assert x.ndim == 2, "Input must have batch dimension."

        sigmas = vp_sde.sigma(t)
        convolved_vars = cov_diag + sigmas[..., None] ** 2
        return -(1 / convolved_vars) * (x - mu[None, :]) * sigmas[:, None]

    vp_sde.score_net = score_net
    return vp_sde


def test_ve_sde(ve_sde):
    ambient_dim = pytest.ambient_dim
    batch_size = 10

    x = torch.cat(
        (
            torch.zeros(batch_size // 2, ambient_dim),
            torch.randn(batch_size // 2, ambient_dim),
        )
    )
    t_1 = torch.linspace(1e-4, 1, batch_size)
    t_0 = 0.5 * t_1

    # Check basic SDE functions have the correct shapes
    assert ve_sde.drift(x, t_0).shape == x.shape
    assert ve_sde.diff(t_0).shape == (batch_size,)
    assert ve_sde.sigma(t_1).shape == (batch_size,)
    assert ve_sde.sigma(t_start=t_0, t_end=t_1).shape == (batch_size,)
    assert ve_sde.score(x, 0).shape == x.shape
    assert ve_sde.score(x, t_0).shape == x.shape

    # Reconstruct a point with the probability flow ODE
    x_end = ve_sde.solve_forward_ode(x, steps=10)
    x_recon = ve_sde.solve_reverse_ode(x_end, steps=10)
    assert x_end.shape == x_recon.shape == x.shape
    assert not x_end.isnan().any()
    assert not x_recon.isnan().any()

    # Reconstruct a point with specific batched timesteps
    x_end = ve_sde.solve_forward_ode(x, t_start=t_0, t_end=t_1, steps=10)
    x_recon = ve_sde.solve_reverse_ode(x_end, t_start=t_1, t_end=t_0, steps=10)
    assert x_end.shape == x_recon.shape == x.shape
    assert not x_end.isnan().any()
    assert not x_recon.isnan().any()

    # Run SDE forward and backward
    x_end_stochastic = ve_sde.solve_forward_sde(x)
    x_recon_stochastic = ve_sde.solve_reverse_sde(x_end)
    assert x_end_stochastic.shape == x.shape
    assert x_recon_stochastic.shape == x.shape
    assert not x_end_stochastic.isnan().any()
    assert not x_recon_stochastic.isnan().any()

    # Run SDE forward and backward with batched timesteps
    print("1", t_0.shape, t_1.shape)
    x_end_stochastic = ve_sde.solve_forward_sde(x, t_start=t_0, t_end=t_1)
    print("2", t_0.shape, t_1.shape)
    print(x_end_stochastic.shape, x_end.shape)
    x_recon_stochastic = ve_sde.solve_reverse_sde(x_end, t_start=t_1, t_end=t_0)
    print("3", t_0.shape, t_1.shape)
    assert x_end_stochastic.shape == x.shape
    assert x_recon_stochastic.shape == x.shape
    assert not x_end_stochastic.isnan().any()
    assert not x_recon_stochastic.isnan().any()


def test_vp_sde(vp_sde):
    ambient_dim = pytest.ambient_dim
    batch_size = 10

    x = torch.cat(
        (
            torch.zeros(batch_size // 2, ambient_dim),
            torch.randn(batch_size // 2, ambient_dim),
        )
    )
    t_1 = torch.linspace(1e-4, 1, batch_size)
    t_0 = 0.5 * t_1

    # Check basic SDE functions have the correct shapes
    assert vp_sde.drift(x, t_0).shape == x.shape
    assert vp_sde.diff(t_0).shape == (batch_size,)
    assert vp_sde.beta(t_1).shape == (batch_size,)
    assert vp_sde.beta_integral(t_0, t_1).shape == (batch_size,)
    assert vp_sde.mu_scale(t_1).shape == (batch_size,)
    assert vp_sde.mu_scale(t_start=t_0, t_end=t_1).shape == (batch_size,)
    assert vp_sde.sigma(t_1).shape == (batch_size,)
    assert vp_sde.sigma(t_start=t_0, t_end=t_1).shape == (batch_size,)
    assert vp_sde.score(x, 0).shape == x.shape
    assert vp_sde.score(x, t_0).shape == x.shape

    # Reconstruct a point with the probability flow ODE
    x_end = vp_sde.solve_forward_ode(x, steps=10)
    x_recon = vp_sde.solve_reverse_ode(x_end, steps=10)
    assert x_end.shape == x_recon.shape == x.shape
    assert not x_end.isnan().any()
    assert not x_recon.isnan().any()

    # Reconstruct a point with specific batched timesteps
    x_end = vp_sde.solve_forward_ode(x, t_start=t_0, t_end=t_1, steps=10)
    x_recon = vp_sde.solve_reverse_ode(x_end, t_start=t_1, t_end=t_0, steps=10)
    assert x_end.shape == x_recon.shape == x.shape
    assert not x_end.isnan().any()
    assert not x_recon.isnan().any()

    # Run SDE forward and backward
    x_end_stochastic = vp_sde.solve_forward_sde(x)
    x_recon_stochastic = vp_sde.solve_reverse_sde(x_end)
    assert x_end_stochastic.shape == x.shape
    assert x_recon_stochastic.shape == x.shape
    assert not x_end_stochastic.isnan().any()
    assert not x_recon_stochastic.isnan().any()

    # Run SDE forward and backward with batched timesteps
    x_end_stochastic = vp_sde.solve_forward_sde(x, t_start=t_0, t_end=t_1)
    x_recon_stochastic = vp_sde.solve_reverse_sde(x_end, t_start=t_1, t_end=t_0)
    assert x_end_stochastic.shape == x.shape
    assert x_recon_stochastic.shape == x.shape
    assert not x_end_stochastic.isnan().any()
    assert not x_recon_stochastic.isnan().any()


def test_vp_generation(vp_sde):
    batch_size = 10

    # Generate and validate samples with the ODE and SDE
    x_noise = torch.randn(batch_size, pytest.ambient_dim)
    x_gen = vp_sde.solve_reverse_ode(x_noise, steps=1000)
    x_gen_stochastic = vp_sde.solve_reverse_sde(x_noise, steps=1000)  # Requires 1k steps
    samples = torch.cat((x_gen, x_gen_stochastic))

    assert not samples.isnan().any()
    assert (samples[:, -2:] <= 0.2).all()  # Last 2 dims have tiny variances
    assert not (samples[:, :-2] <= 0.2).all()  # Rest have larger variances


def test_ve_log_prob(ve_sde):
    batch_size = 10

    x_noise = torch.randn(batch_size, pytest.ambient_dim)
    log_probs = ve_sde.log_prob(x_noise, steps=1000)
    assert log_probs.shape == (batch_size,)


def test_vp_log_prob(vp_sde):
    batch_size = 10

    x_noise = torch.randn(batch_size, pytest.ambient_dim)
    log_probs = vp_sde.log_prob(x_noise, steps=1000)
    assert log_probs.shape == (batch_size,)


# Set different methods for trace estimation with different absolute error tolerances
def test_log_likelihood(get_score_network):
    for samples, t, vpsde in zip(*get_score_network):
        vpsde: VpSde
        log_prob_expected = vpsde.score_net.log_marginal_distribution(
            samples, torch.tensor(t).repeat(samples.shape[0])
        )
        log_prob = vpsde.log_prob(samples, t=t, steps=10)
        relative_errors = torch.abs(log_prob - log_prob_expected) / torch.abs(log_prob_expected)
        assert torch.allclose(
            relative_errors,
            torch.zeros_like(relative_errors),
            atol=0.11 * samples.shape[1] ** 2,
        ), f"Max relative log likelihood errors on time {t}:  {torch.max(relative_errors)}"
