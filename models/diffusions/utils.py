import numbers

import torch


def batch_linspace(start, end, steps):
    """
    Batched linspace function. If start and end are numbers, it will return
    a linspace from start to end of size (steps, ). If start and end are
    torch.Tensors that are of shape (batch_size, ) it will return a batch of
    linspaces from start to end of size (steps, batch_size).

    Args:
        start: (torch.Tensor or float) The start of the linspace.
        end: (torch.Tensor or float) The end of the linspace.
        steps: (int) The number of steps in the linspace.
    Returns:
        (torch.Tensor) The linspace or batch of linspaces.
    """
    # Normal linspace behaviour
    if isinstance(start, numbers.Number) or start.ndim == 0:
        return torch.linspace(start, end, steps)

    # Batched linspace behaviour
    def linspace(start, end):
        return start + torch.arange(0, steps).to(start.device) * (end - start) / (steps - 1)

    return torch.vmap(linspace)(start, end).mT
