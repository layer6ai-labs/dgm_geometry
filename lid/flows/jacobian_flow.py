import functools
import math
from dataclasses import dataclass

import numpy as np
import torch
from nflows.flows.base import Flow
from tqdm import tqdm

from data.transforms.unpack import UnpackBatch
from lid import ModelBasedLIDEstimator
from models.flows import NFlowDiffeomorphism


class JacobianFlowLIDEstimator(ModelBasedLIDEstimator):
    """
    This is a parent class for multiple flow based LID estimators.
    The premise is that the flow is invertible and the Jacobian of the inverse function is calculated,
    and the eigendecomposition of the Jacobian is used to estimate the LID.
    """

    @dataclass
    class Artifact:
        """
        The Jacobian, eigenvectors and eigenvalues of the Jacobian, and the latent representation z
        are stored in this dataclass for further processing.
        """

        jacobian_batch: torch.Tensor
        jtj_eigvals_batch: torch.Tensor
        jtj_eigvecs_batch: torch.Tensor
        z_batch: torch.Tensor

    def __init__(
        self,
        model: Flow,
        use_functorch: bool = True,
        use_vmap: bool = False,  # keep setting
        use_forward_mode: bool = True,
        ambient_dim: int | None = None,
        device: torch.device | None = None,
        unpack: UnpackBatch | None = None,
    ):
        """
        Args:
            model: The flow model that will be used to estimate the LID.
            use_functorch: If True, use the functorch library for jacobian calculation.
            use_vmap: If True, use the vmap function for jacobian calculation.
            use_forward_mode: If True, use forward mode differentiation for jacobian calculation.
            ambient_dim: The dimension of the ambient space.
            device: The device on which the computation will be performed.
            unpack: The unpacking function that will be used to unpack the batched data.
        """
        super().__init__(
            ambient_dim=ambient_dim,
            model=model,
            device=device,
            unpack=unpack,
        )
        self.flow: Flow = self.model
        self.use_functorch = use_functorch
        self.use_vmap = use_vmap
        self.use_forward_mode = use_forward_mode
        self.flow.eval()

    def _encode(
        self,
        x,
        batchwise: bool = True,  # whether the input is batched or not
    ):
        """encode will turn the input x into a latent representation z."""
        if batchwise:
            return self.flow.transform_to_noise(x)
        else:
            return self.flow.transform_to_noise(x.unsqueeze(0)).squeeze(0)

    def _decode(
        self,
        z,
        batchwise: bool = True,  # whether the input is batched or not
    ):
        """decode will turn the latent representation z into a data representation"""
        if batchwise:
            x, _ = self.flow._transform.inverse(z)
            return x
        else:
            x, _ = self.flow._transform.inverse(z.unsqueeze(0))
            return x.squeeze(0)

    @torch.no_grad
    def _preprocess(
        self,
        x: torch.Tensor,
        chunk_size: int = 128,
    ) -> Artifact:
        """Perform the computation necessary for LID estimation.

        Args:
            x: The points at which to estimate LID with shape (batch_size, *).
            verbose: If > 0, display a progress bar.
        """
        # assert if x is not a torch tensor
        assert isinstance(x, torch.Tensor), "x should be a torch.Tensor"

        # all_singular_vals = []
        jtj_eigvals = []
        jtj_eigvecs = []
        jacobian_batch = []
        z_batch = []
        for x_chunk in x.split(chunk_size):
            x_chunk = x.to(self.device)
            z_chunk = self._encode(x_chunk)

            # Calculate the jacobian of the decode function
            if self.use_functorch:
                jac_fn = torch.func.jacfwd if self.use_forward_mode else torch.func.jacrev
                if self.use_vmap:
                    # optimized implementation with vmap, however, it does not work as of yet
                    jac_chunk = torch.func.vmap(
                        jac_fn(functools.partial(self._decode, batchwise=False))
                    )(z_chunk)
                else:
                    jac_chunk = jac_fn(self._decode)(z_chunk)
            else:
                jac_chunk = torch.autograd.functional.jacobian(self._decode, z_chunk)

            # Reshaping the jacobian to be of the shape (batch_size, latent_dim, latent_dim)
            jac_chunk: torch.Tensor
            if self.use_vmap and self.use_functorch:
                jac_chunk = jac_chunk.reshape(
                    z_chunk.shape[0], -1, z_chunk.numel() // z_chunk.shape[0]
                )
            else:
                jac_chunk = jac_chunk.reshape(
                    z_chunk.shape[0], -1, z_chunk.shape[0], z_chunk.numel() // z_chunk.shape[0]
                )
                indices = torch.arange(jac_chunk.shape[0])
                jac_chunk = jac_chunk[indices, :, indices, :]

            jtj = torch.matmul(jac_chunk.transpose(1, 2), jac_chunk)
            jtj = 0.5 * (jtj.transpose(1, 2) + jtj)
            jtj = torch.clamp(jtj, min=-(10**4.5), max=10**4.5)
            jtj = torch.where(jtj.isnan(), torch.zeros_like(jtj), jtj)

            # perform eigendecomposition
            L, Q = torch.linalg.eigh(jtj)

            L = torch.where(L > 1e-20, L, 1e-20 * torch.ones_like(L))

            # move to RAM memory to circumvent overloading the GPU memory
            jtj_eigvals.append(L.cpu())
            jtj_eigvecs.append(Q.cpu())
            jacobian_batch.append(jac_chunk.cpu())
            z_batch.append(z_chunk.cpu())

        return JacobianThresholdEstimator.Artifact(
            jacobian_batch=torch.cat(jacobian_batch),
            jtj_eigvals_batch=torch.cat(jtj_eigvals),
            jtj_eigvecs_batch=torch.cat(jtj_eigvecs),
            z_batch=torch.cat(z_batch),
        )


class JacobianThresholdEstimator(JacobianFlowLIDEstimator):
    """
    The Intrinsic dimension estimator introduced by https://arxiv.org/abs/2403.18910
    and Horvat and Pfister, 2021 https://proceedings.neurips.cc/paper_files/paper/2022/hash/4f918fa3a7c38b2d9b8b484bcc433334-Abstract-Conference.html

    This simply takes the number of singular values of the Jacobian that are more than a threshold as the LID.
    """

    @torch.no_grad
    def compute_lid_from_artifact(
        self,
        lid_artifact: JacobianFlowLIDEstimator.Artifact | None = None,
        singular_value_threshold: float | None = None,
    ) -> torch.Tensor:
        """
        When the singular_value_threshold is None, the LID is calculated as the largest
        gap between the singular values of the Jacobian.

        Otherwise, the LID is calculated as the number of singular values \sigma_i such that

        \sigma_i > exp(2 * singular_value_threshold)

        Therefore, for a very negative singular_value_threshold, the LID will be the ambient dimension.
        If it is very positive, the LID will be 0.
        """
        # count the number of singular values that are more than the threshold
        singular_vals = lid_artifact.jtj_eigvals_batch.to(self.device)
        if singular_value_threshold is None:
            normal_dim = (singular_vals[:, :-1] - singular_vals[:, 1:]).argmax(dim=1) + 1
            lids = (self.ambient_dim - normal_dim).cpu()
        else:
            threshold = math.exp(2 * singular_value_threshold)
            lids = (singular_vals > threshold).sum(dim=1).cpu()

        return lids


class FastFlowLIDL(JacobianFlowLIDEstimator):
    """
    The Intrinsic dimension estimator introduced by a previous version of https://arxiv.org/abs/2403.18910
    here: https://openreview.net/forum?id=jQ596tXT3k
    """

    @torch.no_grad
    def compute_lid_from_artifact(
        self,
        lid_artifact: JacobianFlowLIDEstimator.Artifact | None = None,
        delta: float | None = None,
    ) -> torch.Tensor:
        """
        This uses a LIDL-based formula instead. It takes the derivative of the Gaussian
        convolution. For flows, we don't have the Gaussian convolution, but we can use the
        Jacobian of the inverse function replace the flow with a local linear approximation.

        In which case,

        f(z) = x_0 + J_f(z_0) (z - z_0) + O(||z - z_0||^2)
        where x_0 = f(z_0) and the O(||z - z_0||^2) term is the error term and can be ignored locally.
        f(z) ~ N(x_0, J_f(z_0) J_f(z_0)^T)

        Now the probability density would be Gaussian and can be easily convolved with another Gaussian.
        The Gaussian convolution will be obtained by the following formula:

        \log p(x) * N(0, \sigma^2) \approx \log N(x_0 - J_f(z_0) z_0, J_f(z_0) J_f(z_0)^T + \sigma^2 I)

        We replace \sigma^2 with \exp(2 \delta) and then we take the derivative of the log of the convolution
        with respect to \delta. This will give us the LID.
        """
        assert delta is not None, "The delta parameter should be provided."
        # count the number of singular values that are more than the threshold
        jtj_eigvals_batch = lid_artifact.jtj_eigvals_batch.to(self.device)
        jtj_rot_batch = lid_artifact.jtj_eigvecs_batch.to(self.device)
        z_batch = lid_artifact.z_batch.to(self.device)

        if isinstance(delta, torch.Tensor):
            var = torch.exp(2 * delta)
        else:
            var = np.exp(2 * delta)

        z_transformed = torch.bmm(jtj_rot_batch.transpose(1, 2), z_batch.unsqueeze(-1)).squeeze(-1)
        ret = -torch.sum(1 / (jtj_eigvals_batch + var), dim=1)
        ret = ret + torch.sum(
            jtj_eigvals_batch * (z_transformed / (jtj_eigvals_batch + var)) ** 2, dim=1
        )
        lid_batch = ret * var
        return self.ambient_dim + lid_batch
