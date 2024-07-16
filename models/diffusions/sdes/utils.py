import inspect
import math
import numbers
from contextlib import contextmanager
from typing import Callable, Literal

import torch
from tqdm import tqdm

# A threshold for the dimension of the data, if the dimension is above this threshold, the hutchinson method is used
HUTCHINSON_DATA_DIM_THRESHOLD = 3500


class VpSdeGaussianAnalytical(torch.nn.Module):
    """
    This class represents a score network that
    matches the marginal distributions obtained from
    a VpSde diffusion process mapping isotropic Gaussian
    to a target Gaussian distribution, or `posterior`.

    The posterior is parameterized by a mean and covariance.
    And the beta for the VpSde is a linearly growing function
    from beta_0 to beta_1.

    Although this class is a torch.nn.Module, it is not trained
    and does not contain any parameters. It is used for testing
    our theoretical derivations and implementations in isolation,
    where the missmatch between the trained network and the
    score network would not be a concern anymore.
    """

    def __init__(
        self,
        posterior_mean: torch.Tensor,
        posterior_cov: torch.Tensor,
        beta_min: float = 0.1,
        beta_max: float = 20,
        t_max: float = 1.0,
    ):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.t_max = t_max
        self.posterior_mean = posterior_mean
        self.posterior_cov = posterior_cov

    def beta(self, t):
        return (self.beta_max - self.beta_min) * t / self.t_max + self.beta_min

    def beta_integral(self, t_start, t_end):
        """Integrate beta(t) from t_start to t_end"""
        if not hasattr(self, "beta_diff"):
            self.beta_diff = self.beta_max - self.beta_min
        t_diff = t_end - t_start
        return self.beta_diff / (2 * self.t_max) * (t_end**2 - t_start**2) + self.beta_min * t_diff

    def log_marginal_distribution(self, x: torch.Tensor, t: torch.Tensor):
        """
        Compute the convolution distribution obtained at time 't'
        """

        # if x has more dimensions than 2, assert
        assert len(x.shape) == 2, "x should have shape [batch_size, d]"

        batch_size, d = x.shape
        B_t = self.beta_integral(0, t)
        x_eval = torch.exp(0.5 * B_t)[:, None] * x
        iden = torch.eye(d).to(x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        marginal_cov = self.posterior_cov + (torch.exp(B_t) - 1)[:, None, None] * iden
        # solve the batchwise linear system marginal_cov * y = x_eval
        y = torch.linalg.solve(marginal_cov, (x_eval - self.posterior_mean).unsqueeze(-1)).squeeze(
            -1
        )
        # write the log_prob without torch but analytical
        log_prob = (
            -0.5 * torch.sum((x_eval - self.posterior_mean) * y, dim=1)
            - 0.5 * torch.logdet(marginal_cov)
            - 0.5 * d * torch.log(2 * torch.tensor(torch.pi))
            + 0.5 * d * B_t
        )
        return log_prob

    def log_convolution_distribution(self, x: torch.Tensor, t: torch.Tensor):
        """
        The marginal distribution of the VpSde diffusion process
        """
        B_t = self.beta_integral(0, t)
        d = x.numel() // x.shape[0]
        log_marginal = self.log_marginal_distribution(torch.exp(-B_t / 2)[:, None] * x, t)
        return log_marginal - d * B_t / 2

    def forward(self, x, t):
        """this function would return the score of the VpSde which is the gradient of the log_prob w.r.t. x which is rescaled"""
        # if x has more dimensions than 2, assert
        assert len(x.shape) == 2, f"x should have shape [batch_size, d], but got: {x.shape}"

        batch_size, d = x.shape
        B_t = self.beta_integral(0, t)
        iden = torch.eye(d).to(x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        marginal_cov = self.posterior_cov.to(x.device) + (torch.exp(B_t) - 1)[:, None, None] * iden
        x_eval = torch.exp(0.5 * B_t)[:, None] * x

        # solve the linear system marginal_cov * y = x_eval batchwise
        y = torch.linalg.solve(
            marginal_cov, (x_eval - self.posterior_mean.to(x.device)).unsqueeze(-1)
        ).squeeze(-1)
        rescale_factor = torch.sqrt((1 - torch.exp(-B_t)))
        return -(torch.exp(0.5 * B_t) * rescale_factor)[:, None] * y


def copy_tensor_or_create(t, **kwargs):
    """Returns a copy of the input tensor or creates a new tensor from the input if it is a number."""
    # check if t is a number or not
    if isinstance(t, numbers.Number):
        return torch.tensor(t, **kwargs)
    elif isinstance(t, torch.Tensor):
        return t.clone().detach()
    else:
        raise ValueError(f"Cannot copy object of type {type(t)}")


def filter_kwargs_for_function(func, **kwargs):
    # Get the signature of the function
    sig = inspect.signature(func)
    # Extract parameter names from the function signature
    param_names = set(sig.parameters.keys())
    # Filter kwargs to only include keys that match the function's parameter names
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in param_names}
    # Call the function with the filtered kwargs
    return filtered_kwargs


def _jvp_mode(flag: bool, device: torch.device):
    """
    Flags that need to be set for jvp to work with attention layers.

    NOTE: This has been tested on torch version 2.1.1, hopefully,
    this issue will be resolved in a future version of torch
    as jvp mode reduces the speed of JVP computation.
    """
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(not flag)
        torch.backends.cuda.enable_mem_efficient_sdp(not flag)
        torch.backends.cuda.enable_math_sdp(flag)


@contextmanager
def _jvp_mode_enabled(device: torch.device):
    _jvp_mode(True, device)
    try:
        yield
    finally:
        _jvp_mode(False, device)


def compute_trace_of_jacobian(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    method: Literal["hutchinson_gaussian", "hutchinson_rademacher", "deterministic"] | None = None,
    hutchinson_sample_count: int = HUTCHINSON_DATA_DIM_THRESHOLD,
    chunk_size: int = 128,
    seed: int = 42,
    verbose: bool = False,
):
    """
    fn is a function mapping \R^d to \R^d, this function computes the trace of the Jacobian of fn at x.

    To do so, there are different methods implemented:

    1. The Hutchinson estimator:
        This is a stochastic estimator that uses random vector to estimate the trace.
        These random vectors can either come from the Gaussian distribution (if method=`hutchinson_gaussian` is specified)
        or from the Rademacher distribution (if method=`hutchinson_rademacher` is specified).
    2. The deterministic method:
        This is not an estimator and computes the trace by taking all the x.dim() canonical basis vectors times $\sqrt{d}$ (?)
        and taking the average of their quadratic forms. For data with small dimension, the deterministic method
        is the best.

    The implementation of all of these is as follows:
        A set of vectors of the same dimension as data are sampled and the value [v^T \\nabla_x v^T fn(x)] is
        computed using jvp. Finally, all of these values are averaged.

    Args:
        fn (Callable[[torch.Tensor], torch.Tensor]):
            A function that takes in a tensor of size [batch_size, *data_shape] and returns a tensor of size [batch_size, *data_shape]
        x (torch.Tensor): a batch of inputs [batch_size, input_dim]
        method (str, optional):
            chooses between the types of methods to evaluate trace.
            it defaults to None, in which case the most appropriate method is chosen based on the dimension of the data.
        hutchinson_sample_count (int):
            The number of samples for the stochastic methods, if deterministic is chosen, this is ignored.
        chunk_size (int):
            Jacobian vector products can be done in parallel for better speed-up, this is the size of the parallel batch.
    Returns:
        traces (torch.Tensor): A tensor of size [batch_size,] where traces[i] is the trace computed for the i'th batch of data
    """
    # use seed to make sure that the same random vectors are used for the same data
    # NOTE: maybe creating a fork of the random number generator is a better idea here!
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        # save batch size and data dimension and shape
        batch_size = x.shape[0]
        data_shape = x.shape[1:]
        ambient_dim = x.numel() // x.shape[0]
        if ambient_dim > HUTCHINSON_DATA_DIM_THRESHOLD:
            method = method or "hutchinson_gaussian"
        else:
            method = method or "deterministic"

        # The general implementation is to compute the quadratic forms of [v^T \\nabla_x v^T score(x, t)] in a list and then take the average
        all_quadratic_forms = []
        sample_count = hutchinson_sample_count if method != "deterministic" else ambient_dim
        # all_v is a tensor of size [batch_size * sample_count, *data_shape] where each row is an appropriate vector for the quadratic forms
        if method == "hutchinson_gaussian":
            all_v = torch.randn(size=(batch_size * sample_count, *data_shape)).cpu().float()
        elif method == "hutchinson_rademacher":
            all_v = (
                torch.randint(size=(batch_size * sample_count, *data_shape), low=0, high=2)
                .cpu()
                .float()
                * 2
                - 1.0
            )
        elif method == "deterministic":
            all_v = torch.eye(ambient_dim).cpu().float() * math.sqrt(ambient_dim)
            # the canonical basis vectors times sqrt(d) the sqrt(d) coefficient is applied so that when the
            # quadratic form is computed, the average of the quadratic forms is the trace rather than their sum
            all_v = all_v.repeat_interleave(batch_size, dim=0).reshape(
                (batch_size * sample_count, *data_shape)
            )
        else:
            raise ValueError(f"Method {method} for trace computation not defined!")
        # x is also duplicated as much as needed for the computation
        all_x = (
            x.cpu()
            .unsqueeze(0)
            .repeat(sample_count, *[1 for _ in range(x.dim())])
            .reshape(batch_size * sample_count, *data_shape)
        )

        all_quadratic_forms = []
        rng = list(zip(all_v.split(chunk_size), all_x.split(chunk_size)))
        # compute chunks separately
        rng = tqdm(rng, desc="Computing the quadratic forms") if verbose else rng
        idx_dbg = 0
        with _jvp_mode_enabled(x.device):
            for vx in rng:
                idx_dbg += 1

                v_batch, x_batch = vx
                v_batch = v_batch.to(x.device)
                x_batch = x_batch.to(x.device)

                all_quadratic_forms.append(
                    torch.sum(
                        v_batch * torch.func.jvp(fn, (x_batch,), tangents=(v_batch,))[1],
                        dim=tuple(range(1, x.dim())),
                    ).cpu()
                )
    # concatenate all the chunks
    all_quadratic_forms = torch.cat(all_quadratic_forms)
    # reshape so that the quadratic forms are separated by batch
    all_quadratic_forms = all_quadratic_forms.reshape((sample_count, x.shape[0]))
    # take the average of the quadratic forms for each batch
    return all_quadratic_forms.mean(dim=0).to(x.device)
