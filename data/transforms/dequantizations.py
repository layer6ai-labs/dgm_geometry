import torch


class DequantizeTransform:
    """
    Takes in a tensor that has values in the range [0, 1] but equally spaced taking
    'num_vals' values. This transform adds uniform noise that does not change the
    ordering of the values and then scales the values back to the range [0, 1] but
    makes it more continuous.

    It is appropritate for normalizing flows.
    """

    def __init__(self, num_vals: int = 256):
        self.num_vals = num_vals

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x * (self.num_vals - 1) + torch.rand_like(x)) / self.num_vals


class QuantizeTransform:
    """
    Performs the inverse operation of DequantizeTransform.
    """

    def __init__(self, num_vals: int = 256):
        self.num_vals = num_vals

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.num_vals).floor() / (self.num_vals - 1)
