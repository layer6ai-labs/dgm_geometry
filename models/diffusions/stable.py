"""
A group of objects designed to load Stable Diffusion models and interface them with the
rest of the codebase Stable Diffusion models in the codebase.
"""

import torch
import torch.nn as nn

from .networks import DiffusersDDPMWrapper
from .sdes import VpSde


class StableDiffusionScoreNet(DiffusersDDPMWrapper):
    def __init__(self, pipeline):
        super().__init__(pipeline.unet, t_factor=pipeline.scheduler.config.num_train_timesteps)
        self.empty_str_embedding = StableDiffusionPromptEncoder(pipeline)("")

    def forward(self, x, t, encoder_hidden_states=None, **kwargs):
        if encoder_hidden_states is None:
            encoder_hidden_states = self.empty_str_embedding.repeat(x.shape[0], 1, 1)
        return super().forward(x, t, encoder_hidden_states=encoder_hidden_states, **kwargs)


class StableDiffusionSde(VpSde):
    """Create a VpSDE from a diffusers pipeline containing a pretrained model"""

    def __init__(self, pipeline):
        self.pipeline = pipeline

        num_train_timesteps = pipeline.scheduler.config.num_train_timesteps
        assert num_train_timesteps == 1000
        beta_min = pipeline.scheduler.config.beta_start * num_train_timesteps
        beta_max = pipeline.scheduler.config.beta_end * num_train_timesteps

        # Use the embedding of the empty str as a default when score net is not passed
        # any other conditioning input
        score_net = StableDiffusionScoreNet(self.pipeline)
        super().__init__(score_net=score_net, beta_min=beta_min, beta_max=beta_max)


class StableDiffusionEncoder(nn.Module):
    """A transform that represents encoding an image using a pipeline's VAE"""

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.vae = pipeline.vae

    def __call__(self, x):
        z = self.vae.encode(x).latent_dist.mean
        return z * self.vae.config.scaling_factor


class StableDiffusionDecoder(nn.Module):
    """A transform that represents decoding an image using a pipeline's VAE"""

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.vae = pipeline.vae

    def __call__(self, z):
        z_scaled = z / self.vae.config.scaling_factor
        x = self.vae.decode(z_scaled).sample
        return self.pipeline.image_processor.postprocess(x)


class StableDiffusionPromptEncoder(nn.Module):
    def __init__(self, pipeline, device=None):
        super().__init__()
        self.pipeline = pipeline
        self.encoder = self.pipeline.text_encoder

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def __call__(self, prompt):
        prompt_embeds, neg_prompt_embeds = self.pipeline.encode_prompt(
            prompt,
            self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return prompt_embeds

    def to(self, device):
        self.device = device
        super().to(device)
        return self
