import torch


class UnflattenTransform:
    """
    An unflattening transform that can be written in the configurations
    """

    def __init__(self, shape):
        self.shape = []
        for s in shape:
            self.shape.append(s)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.reshape(x, (x.shape[0], *self.shape))
