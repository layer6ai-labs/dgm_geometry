import math
from typing import List, Tuple

import torch
import torch.nn as nn
from diffusers import UNet2DModel as DiffusersUNet2D


# taken from diffusers
def _get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class SimpleMLP(nn.Module):
    """Simple MLP for flat data"""

    def __init__(self, in_dim, out_dim, hidden_sizes=(32, 32)):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_sizes = hidden_sizes

        layers = []
        prev_size = in_dim
        for size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.SiLU())
            prev_size = size

        layers.append(nn.Linear(prev_size, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleDiffusionMLP(SimpleMLP):
    def __init__(self, data_dim, hidden_sizes=(32, 32)):
        super().__init__(in_dim=data_dim + 1, out_dim=data_dim, hidden_sizes=hidden_sizes)
        self.sample_size = 1

    def forward(self, x, t):
        t = t.to(x.device)
        return self.net(torch.cat([x, t[..., None]], dim=-1))


class MLPUnet(nn.Module):
    """
    This module represents a Unet hierarchical structure
    but for flat data.
    """

    def __init__(
        self,
        data_dim,
        hidden_sizes=(32, 32),
        time_embedding_dim: int | None = None,
    ):
        super().__init__()
        self.layers = []
        self.time_embedding_dim = time_embedding_dim or 1
        self.layer_info: List[Tuple[int, int]] = [(self.time_embedding_dim + data_dim, -1)]
        for size in hidden_sizes:
            self.layer_info.append((size, -1))
        ref_layer = len(self.layer_info) - 1
        for size in hidden_sizes[::-1][1:]:
            ref_layer -= 1
            self.layer_info.append((size, ref_layer))
        self.layer_info.append((data_dim, 0))
        for i in range(1, len(self.layer_info)):
            layer_sz, ref_layer_idx = self.layer_info[i]
            last_layer_sz, _ = self.layer_info[i - 1]
            if ref_layer_idx == -1:
                self.layers.append(nn.Linear(last_layer_sz, layer_sz))
            else:
                self.layers.append(
                    nn.Linear(last_layer_sz + self.layer_info[ref_layer_idx][0], layer_sz)
                )
        self.layers = nn.ModuleList(self.layers)
        self.activation = nn.SiLU()

    def forward(self, x, t):
        if self.time_embedding_dim > 1:
            t = _get_timestep_embedding(t, self.time_embedding_dim)

        embeddings = [torch.cat([x, t], dim=-1)]
        for i, layer in enumerate(self.layers):
            if self.layer_info[i + 1][1] == -1:
                interim = layer(embeddings[-1])
            else:
                interim = layer(
                    torch.cat([embeddings[self.layer_info[i + 1][1]], embeddings[-1]], dim=-1)
                )
            if i < len(self.layers) - 1:
                interim = self.activation(interim)

            embeddings.append(interim)
        return embeddings[-1]


# NOTE: using this score network for tabular data is not recommended!
class AttnScoreNetwork(nn.Module):
    def __init__(self, d, k, L, num_heads, dim_feedforward):
        super(AttnScoreNetwork, self).__init__()
        self.d = d
        self.k = k
        self.L = L

        # Embedding layer
        self.embedding = nn.Linear(self.d, self.d * k)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=k,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    batch_first=True,
                )
                for _ in range(L)
            ]
        )

        # Projection back to original dimension (d)
        self.projection = nn.Linear(k, 1)

    def forward(self, x, t):
        # x: [batch_size, d]
        batch_size = x.shape[0]

        # Embedding
        x = self.embedding(x)  # [batch_size, d * k]

        t = _get_timestep_embedding(t, self.k)
        x = torch.cat([x, t], dim=-1)  # [batch_size, d * k + k]

        x = x.reshape(batch_size, self.d + 1, self.k)  # [batch_size, d + 1, k]

        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)

        # Project each k-dimensional output back to 1 dimension
        x = self.projection(x)  # [batch_size, d + 1, k] -> [batch_size, d + 1, 1]

        return x.squeeze(-1)[:, :-1]  # [batch_size, d]


class UNet2D(DiffusersUNet2D):
    """Class adapter for diffusers UNet.

    Diffusers networks return a UNet2dOutput data class on the forward pass rather than a
    simple tensor; this makes them awkward to use, so we adapt them to return the tensor
    directly.
    """

    def __init__(self, *args, t_factor=1000, **kwargs):
        self.t_factor = t_factor
        super().__init__(*args, **kwargs)

    def forward(self, x, t, **kwargs):
        t = t * self.t_factor  # Preprocessing for timestep embedding
        return super().forward(sample=x, timestep=t, **kwargs).sample


class DiffusersDDPMWrapper(nn.Module):
    """Wrapper for UNets that have been trained "DDPM-style" as opposed to "SDE-style."

    This class is used to load pretrained DDPM-style UNets. The UNet used to estimate
    epsilon in the DDPM formulation of diffusion (Ho et al., 2020) is equivalent to
    - sigma * score  in the score-based formulation (Song et al., 2020) (where
    sigma * score = score_net). This class is used to port the former into the latter
    style (which this codebase is constructed around).

    (Aside from the sign flip, the other salient change when porting a UNet is that the
    "betas" of DDPM-style UNets need to be multipled by the number of timesteps used to
    train the DDPM when passed into the SDE. Ie., beta_SDE = T * beta_DDPM. See page 14
    of Song et al. (2020).)

    Note also the difference between this class (a "wrapper"), and the UNet2d class
    (an "adapter"), which use composition and inheritance respectively to interface
    with external UNet code.
    """

    def __init__(self, unet, t_factor=1000):
        super().__init__()
        self.unet = unet
        self.t_factor = t_factor

    def forward(self, x, t, **kwargs):
        t = t * self.t_factor  # Preprocessing for timestep embedding
        return -self.unet(sample=x, timestep=t, **kwargs).sample
