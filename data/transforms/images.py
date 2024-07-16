import numpy as np
import torch
from PIL import Image


class DiscardAlphaChannel:
    """If an image has an alpha (4th) channel, remove it.

    This is useful for web-scale datasets; many images on the web
    have alpha.
    """

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            return img[:3, :, :]
        elif isinstance(img, Image.Image):
            img = np.array(img)
            img = img[:, :, :3]
            return Image.fromarray(img)
        else:
            raise TypeError("Input should be a PIL Image or a Tensor")
