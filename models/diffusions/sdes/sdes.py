import functools
import math
import numbers
from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn
from tqdm import tqdm

from ..utils import batch_linspace
from .utils import HUTCHINSON_DATA_DIM_THRESHOLD, compute_trace_of_jacobian, copy_tensor_or_create


class Sde(ABC, nn.Module):
    """The forward- and reverse-SDEs defining a diffusion model.

    This closely follows the math described in Song et al. (2020).  (Available here
    https://arxiv.org/abs/2011.13456). Equation numbers in the comments throughout
    this file refer to the equations in the paper.

    This class implements a general SDE, as given by equation (5),
        dx = f(x, t)dt + g(t)dw,
    and an approximation of its reverse SDE, the true form of which is given by equation (6):
        dx = [f(x, t) - g(t)^2 grad_x log p_t(x)]dt + g(t) dw',
    where w' is a reverse Brownian motion.

    In reality, we approximate grad_x log p_t(x) with our score network, s(x, t).
    Specific choices of f(x, t) and g(t) should be implemented as subclasses.

    This code implements:
    (1) The forward and backward SDE.
    (2) The forward and backward ODE.
    (3) Log probability estimates.

    Args:
        score_net: a network corresponding to (x, t) |-> s(x, t) * sigma(t), with sigma(t)
            as defined in self.sigma(t) below
    """

    def __init__(self, score_net: torch.nn.Module):
        super().__init__()
        self.score_net = score_net

    @abstractmethod
    def drift(self, x, t):
        """The drift coefficient f(x, t) of the forward SDE"""

    @abstractmethod
    def diff(self, t):
        """The diffusion coefficient g(t) of the forward SDE"""

    @abstractmethod
    def sigma(self, t_end, t_start=0):
        """The standard deviation of x(t_end) | x(t_start)"""

    def score(self, x, t, **score_kwargs):
        """
        The score s(x, t) of the forward SDE at time t,
        the output of the model is used directly, otherwise, the output is normalized by the standard deviation of the score.
        """

        sigma_t = self.sigma(t).to(x.device)

        t = copy_tensor_or_create(t, device=x.device)

        if t.ndim == 0:
            t = t.expand(x.shape[0])  # t should be batched for the score_net
        else:
            new_dims = x.ndim - sigma_t.ndim  # Expand sigma_t for broadcasting
            sigma_t = sigma_t.reshape(x.shape[:1] + (1,) * new_dims)

        # Compute score
        score_sigma_t = self.score_net(x, t, **score_kwargs)
        return score_sigma_t / sigma_t

    @abstractmethod
    def solve_forward_sde(self, x_start, t_end=1.0, t_start=0.0, return_eps=False):
        """Expectation: t_start < t_end"""

    def solve_forward_ode(self, x_start, t_start=1e-4, t_end=1.0, steps=1000, **score_kwargs):
        t_start, t_end = self._match_timestep_shapes(t_start, t_end)
        assert torch.all(t_start <= t_end)

        return self._solve(x_start, t_start, t_end, steps, stochastic=False, **score_kwargs)

    def solve_reverse_sde(self, x_start, t_start=1.0, t_end=1e-4, steps=1000, **score_kwargs):
        t_start, t_end = self._match_timestep_shapes(t_start, t_end)
        assert torch.all(t_start >= t_end)

        return self._solve(x_start, t_start, t_end, steps, stochastic=True, **score_kwargs)

    def solve_reverse_ode(self, x_start, t_start=1.0, t_end=1e-4, steps=1000, **score_kwargs):
        t_start, t_end = self._match_timestep_shapes(t_start, t_end)
        assert torch.all(t_start >= t_end)

        return self._solve(x_start, t_start, t_end, steps, stochastic=False, **score_kwargs)

    @staticmethod
    def _match_timestep_shapes(t_start, t_end):
        t_start = copy_tensor_or_create(t_start)
        t_end = copy_tensor_or_create(t_end)
        if t_start.ndim > t_end.ndim:
            t_end = torch.full_like(t_start, fill_value=t_end)
        elif t_start.ndim < t_end.ndim:
            t_start = torch.full_like(t_end, fill_value=t_start)
        return t_start, t_end

    @torch.no_grad()
    def _solve(
        self,
        x_start: torch.Tensor,
        t_start: float = 1.0,
        t_end: float = 1e-4,
        steps: int = 1000,
        stochastic: bool = True,
        **score_kwargs,
    ):
        """Solve the SDE or ODE with an Euler(-Maruyama) solver.

        Note that this can be used for either the forward or backward solve, depending on whether
        t_start < t_end (forward) or t_start > t_end (reverse). Note that this method is not
        appropriate for the forward SDE; the forward SDE should have an analytical solution.

        TODO: Add predictor-corrector steps.

        Args:
            x_start (Tensor of shape (batch_size, ...)): The starting point
            t_start: The starting time
            t_end: The final time (best not set to zero for numerical stability)
            steps: The number of steps for the solver
            stochastic: Whether to use the SDE (True) or ODE (False)

        Returns:
            x_end: (Tensor of shape (batch_size, ...))
        """
        device = x_start.device
        x = x_start.detach().clone()

        ts = batch_linspace(t_start, t_end, steps=steps).to(device)
        delta_t = copy_tensor_or_create((t_end - t_start) / (steps - 1))  # Negative in reverse time

        for t in ts:
            score = self.score(x, t, **score_kwargs)
            drift = self.drift(x, t)
            diff = self.diff(t)

            if t.ndim > 0:  # diff is batched, so add dimensions for broadcasting
                new_dims = x.ndim - t.ndim
                diff = diff.reshape(x.shape[:1] + (1,) * new_dims)
                delta_t = delta_t.reshape(x.shape[:1] + (1,) * new_dims)

            if stochastic:
                # Perform an Euler-Maruyama step on the reverse SDE from equation (6)
                delta_w = delta_t.abs().sqrt() * torch.randn(x.shape).to(device)
                dx = (drift - diff**2 * score) * delta_t + diff * delta_w
            else:
                # Compute an Euler step on the reverse ODE from equation (13)
                dx = (drift - diff**2 * score / 2) * delta_t

            x += dx
        return x

    def _trace_of_drift_derivative(
        self,
        x: torch.Tensor,
        t: float,
        # Set by default to the less efficient but accurate method:
        method: (
            Literal["hutchinson_gaussian", "hutchinson_rademacher", "deterministic"] | None
        ) = None,
        # The number of samples if one opts for estimation methods to save time:
        hutchinson_sample_count: int = HUTCHINSON_DATA_DIM_THRESHOLD,
        chunk_size: int = 128,
        seed: int = 42,
        verbose: bool = False,
    ):
        """
        Return the trace of the drift derivative for the log_prob calculation.
        In the generic case, we can use the Hutchinson estimator for this purpose.
        However, in many cases such as VpSDE and VeSDE, the trace of the drift derivative
        can be directly computed using the diffusion hyperparameters.

        For example,

        VP-SDE: this value is \\beta(t) \\times d where d is the dimension of the data.
        VE-SDE: this value is 0.
        """
        drift_fn = functools.partial(self.drift, t=t)
        return compute_trace_of_jacobian(
            fn=drift_fn,
            x=x,
            method=method,
            hutchinson_sample_count=hutchinson_sample_count,
            chunk_size=chunk_size,
            seed=seed,
            verbose=verbose,
        )

    def laplacian(
        self,
        x: torch.Tensor,
        t: float,
        # Set by default to the less efficient but accurate method:
        method: (
            Literal["hutchinson_gaussian", "hutchinson_rademacher", "deterministic"] | None
        ) = None,
        # The number of samples if one opts for estimation methods to save time:
        hutchinson_sample_count: int = HUTCHINSON_DATA_DIM_THRESHOLD,
        chunk_size: int = 128,
        seed: int = 42,
        verbose: bool = False,
        **score_kwargs,
    ):
        """
        This function computes the Laplacian for a given batch of datapoints. Laplacian is the trace of the hessian
        of the density (log_prob) function. The Hessian of the log_prob is in fact the Jacbian of the score function
        therefore, we can compute the trace of the Jacobian of the score function instead. Thus, we only pass
        the score function to the JVP-based trace estimator `compute_trace_of_jacobian`.

        For more information, refer to the documention of `.utils.compute_trace_of_jacobian`.
        """

        # score_fn takes in 'x' and returns the Jacobian of the density for 'x'
        score_fn = functools.partial(self.score, t=t, **score_kwargs)
        return compute_trace_of_jacobian(
            fn=score_fn,
            x=x,
            method=method,
            hutchinson_sample_count=hutchinson_sample_count,
            chunk_size=chunk_size,
            seed=seed,
            verbose=verbose,
        )

    @torch.no_grad()
    def log_prob(
        self,
        x: torch.Tensor,
        t: float = 1e-4,
        t_end: float = 1.0,
        steps: int = 1000,
        verbose: bool = False,
        drift_trace_kwargs: dict = None,
        laplacian_kwargs: dict = None,
        shared_trace_kwargs: dict = None,
        **score_kwargs,
    ):
        """
        For the marginal density at time 't', this function computes the log probability of the batch of data
        in the input 'x'.  This is done by first rewriting the SDE in the ODE format and then using the instantaneous
        change-of-variables formulaton of the log_probabilities.

        Args:
            x: a batch of input tensors
            t: the timestep of the ODE.
            t_end: the final time of the ODE.
            steps: The number of steps for the ODE solver
            verbose: If True, shows a progress-bar of the ODE as it is being solved
            drift_trace_kwargs: Keyword arguments to be passed into
                `self._trace_of_drift_derivative` for trace estimation
            laplacian_kwargs: Keyword arguments to be passed into `self.laplacian` for trace
                estimation
            shared_trace_kwargs: Keyword arguments to be passed to be trace estimation methods;
                will be overriden by any of the two above dictionaries
            **score_kwargs: Keyword arguments to be passed into the score function
        Returns:
            A tensor of size (batch_size, ) with the i'th element being the corresponding Gaussian convolution.
        """
        assert t <= t_end, f"t should be less than t_end, got t={t} and t_end={t_end}"

        x = x.clone().detach()
        device = x.device
        batch_size = x.shape[0]

        if drift_trace_kwargs is None:
            drift_trace_kwargs = {}

        if laplacian_kwargs is None:
            laplacian_kwargs = {}

        if shared_trace_kwargs is None:
            shared_trace_kwargs = {}

        # Create a tensor of zeros with the same device and dtype as 'x' with shape (batch_size,)
        log_p = torch.zeros(batch_size, device=device, dtype=x.dtype)

        ts = batch_linspace(t, t_end, steps=steps).to(device)
        delta_s = copy_tensor_or_create((t_end - t) / (steps - 1))
        rng = tqdm(ts, desc="Iterating the ODE") if verbose else ts

        for s in rng:
            # The ODE is:
            # dx = f(x, t) dt - 0.5 * g^2(t) \\nabla_x log p_t(x) dt
            # the instantaneous change of variables formula takes the derivative of the RHS and
            # computes its trace to update the log probability
            trace_of_drift_derivative = self._trace_of_drift_derivative(
                x=x, t=s, **(shared_trace_kwargs | drift_trace_kwargs)
            )
            trace_of_score_derivative = self.laplacian(
                x=x, t=s, **(shared_trace_kwargs | laplacian_kwargs | score_kwargs)
            )
            # trace is a linear function so we can separate out the different trace terms
            log_p += delta_s * (
                trace_of_drift_derivative - 0.5 * self.diff(s) ** 2 * trace_of_score_derivative
            )
            # Update the value of x using the ODE appropriately (Euler)
            x += delta_s * (
                self.drift(x, s) - 0.5 * self.diff(s) ** 2 * self.score(x, s, **score_kwargs)
            )

        # finally add the prior
        log_p += self.prior_log_prob(x, t_end)

        return log_p

    def prior_log_prob(self, x: torch.Tensor, t_end: float = 1.0):
        """In diffusion models, the prior is assumed to be a Gaussian with mean 0 and standard deviation sigma(t_end)"""
        sigma = self.sigma(t_end)
        ambient_dim = x.numel() // x.shape[0]
        # return the log probability of a Gaussian with mean 0 and standard deviation sigma
        log_normalizing_factor = 0.5 * ambient_dim * math.log(2 * math.pi * sigma**2)
        exponential_term = -0.5 * torch.sum(x * x, dim=tuple(range(1, x.dim()))) / sigma**2
        return exponential_term - log_normalizing_factor


class VpSde(Sde):
    """The variance-preserving SDE described by Song et al. (2020) in equation (11).

    Here, the SDE is given by
    dx = -(1/2) beta(t) x dt + sqrt(beta(t)) dw;
    ie., f(x, t) = -(1/2)beta(t)x and g(t) = sqrt(beta(t)).

    For beta(t), we use a linear schedule: beta(t) = (beta_max - beta_min)*(t/T) + beta_min.
    """

    def __init__(
        self,
        score_net: torch.nn.Module,
        beta_min: float = 0.1,
        beta_max: float = 20,
        t_max: float = 1.0,
    ):
        self.beta_min = torch.tensor(beta_min)
        self.beta_max = torch.tensor(beta_max)
        self.t_max = torch.tensor(t_max)
        super().__init__(score_net)

    def drift(self, x, t):
        """The drift coefficient f(x, t) of the forward SDE"""
        t = copy_tensor_or_create(t, device=x.device)
        if t.ndim > 0:  # Add dimensions for broadcasting
            new_dims = x.ndim - t.ndim
            t = t.reshape(x.shape[:1] + (1,) * new_dims)

        return -self.beta(t) * x / 2

    def diff(self, t):
        """The diffusion coefficient g(t) of the forward SDE"""
        return torch.sqrt(self.beta(t))

    def mu_scale(self, t_end, t_start=0.0):
        """Scaling factor for the mean of x(t_end) | x(t_start).

        The mean should equal mu_scale(t_end, t_start) * x(t_start)
        """
        return torch.exp(-self.beta_integral(t_start, t_end) / 2)

    def sigma(self, t_end, t_start=0):
        """The standard deviation of x(t_end) | x(t_start)"""
        return torch.sqrt(1.0 - torch.exp(-self.beta_integral(t_start, t_end)))

    def beta(self, t):
        return (self.beta_max - self.beta_min) * t / self.t_max + self.beta_min

    def beta_integral(self, t_start, t_end):
        """Integrate beta(t) from t_start to t_end"""
        if not hasattr(self, "beta_diff"):
            self.beta_diff = self.beta_max - self.beta_min
        t_diff = t_end - t_start
        return self.beta_diff / (2 * self.t_max) * (t_end**2 - t_start**2) + self.beta_min * t_diff

    def _trace_of_drift_derivative(self, x: torch.Tensor, t: float):
        """Override the trace of the drift derivative into an anlytical form which is fast to compute"""
        ambient_dim = x.numel() // x.shape[0]
        batch_size = x.shape[0]
        return (
            -0.5
            * torch.ones(batch_size, device=x.device, dtype=x.dtype)
            * ambient_dim
            * self.beta(t)
        )

    def solve_forward_sde(self, x_start, t_end=1.0, t_start=0.0, return_eps=False):
        """Solve the SDE forward from time t_start to t_end"""
        t_start, t_end = self._match_timestep_shapes(t_start, t_end)
        t_start, t_end = t_start.to(x_start.device), t_end.to(x_start.device)
        assert torch.all(t_start <= t_end)

        mu_scale = self.mu_scale(t_start=t_start, t_end=t_end)
        sigma_end = self.sigma(t_start=t_start, t_end=t_end)
        eps = torch.randn_like(x_start)

        if mu_scale.ndim > 0:  # Add a broadcasting dimensions to the scalars
            new_dims = x_start.ndim - mu_scale.ndim
            mu_scale = mu_scale.reshape(x_start.shape[:1] + (1,) * new_dims)
            sigma_end = sigma_end.reshape(x_start.shape[:1] + (1,) * new_dims)

        x_end = mu_scale * x_start + sigma_end * eps

        if return_eps:  # epsilon, the random noise value, may be needed for training
            return x_end, eps
        else:
            return x_end


class VeSde(Sde):
    """The variance-exploding SDE described by Song et al. (2020) in equation (9).

    Here, the SDE is given by
    dx = sqrt(d(sigma^2(t))/dt) dw

    We use a geometric schedule: sigma(t) = sigma_min * (sigma_max / sigma_min)^t.
    """

    def __init__(
        self,
        score_net: torch.nn.Module,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        t_max: float = 1.0,
    ):
        self.sigma_min = torch.tensor(sigma_min)
        self.sigma_max = torch.tensor(sigma_max)
        self.t_max = t_max
        super().__init__(score_net)

    def drift(self, x, t):
        """The drift coefficient f(x, t) of the forward SDE"""
        return torch.zeros_like(x)

    def diff(self, t):
        """The diffusion coefficient g(t) of the forward SDE

        Here, this is given by sqrt(d(sigma^2(t))/dt).
        """
        sigma = self.sigma(t)
        diff = sigma * torch.sqrt(2 * (torch.log(self.sigma_max) - torch.log(self.sigma_min)))
        return diff

    def sigma(self, t_end, t_start=0.0):
        """The standard deviation of x(t_end) | x(t_start)"""
        if isinstance(t_start, numbers.Number) and t_start == 0:  # t_start equals the number 0
            return self.sigma_min * (self.sigma_max / self.sigma_min) ** t_end
        else:
            return torch.sqrt(self.sigma(t_end) ** 2 - self.sigma(t_start) ** 2)

    def _trace_of_drift_derivative(self, x: torch.Tensor, t: float):
        """Override the trace of the drift derivative into an anlytical form which is fast to compute"""
        batch_size = x.shape[0]
        return torch.zeros(batch_size, device=x.device, dtype=x.dtype)

    def solve_forward_sde(self, x_start, t_end=1.0, t_start=0.0, return_eps=False):
        """Solve the SDE forward from time t_start to t_end"""
        t_start, t_end = self._match_timestep_shapes(t_start, t_end)
        t_start, t_end = t_start.to(x_start.device), t_end.to(x_start.device)
        assert torch.all(t_start <= t_end)

        sigma_end = self.sigma(t_start=t_start, t_end=t_end)
        eps = torch.randn_like(x_start)

        if sigma_end.ndim > 0:  # sigma is batched, so add a dim for broadcasting
            sigma_end = sigma_end[..., None]

        x_end = sigma_end * torch.randn_like(x_start)

        if return_eps:  # epsilon, the random noise value, may be needed for training
            return x_end, eps
        else:
            return x_end
