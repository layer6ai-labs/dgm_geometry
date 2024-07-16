import numpy as np
import torch
import torch.distributions as dist


class Mondrian(dist.Distribution):
    def __init__(self, height=32, width=32):
        self.height = height
        self.width = width

    def sample(self, sample_shape):
        arr = torch.zeros((np.prod(sample_shape), 3, self.height, self.width))
        for i, _ in enumerate(arr):
            arr[i] = self.random_color_image(self.height, self.width)
        return arr.reshape(sample_shape + (3, self.height, self.width))

    @staticmethod
    def random_color_image(height, width):
        """Sample a random mondrian image."""
        img = torch.zeros((height, width, 3))
        row = torch.randint(low=4, high=height - 4, size=(1,)).item()
        column = torch.randint(low=4, high=width - 4, size=(1,)).item()

        colors = torch.rand(size=(4, 3))
        img[:row, :column, :] = colors[0, :]
        img[row + 1 :, :column, :] = colors[1, :]
        img[:row, column + 1 :, :] = colors[2, :]
        img[row + 1 :, column + 1 :, :] = colors[3, :]

        img[row, :, :] = torch.rand(size=(32, 3))
        img[:, column, :] = torch.rand(size=(32, 3))

        return torch.permute(img, (2, 0, 1))
