import torch


class LogitTransform:
    """
    This transform changes a tensor in the range [0, 1] to the range [-inf, inf] using the logit function.
    The logit function is actually the inverse of the sigmoid function and is an appropriate data transform
    to deal with images when modelling them with normalizing flows.

    The logit function is defined as:

    logit(x) = log(x_tilde / (1 - x_tilde))

    where x_tilde = alpha + (1 - alpha) * x

    alpha and eps are hyperparameters that ensure that the logit function does not blow up to -inf
    or +inf.
    """

    def __init__(self, alpha=0.05, eps=1e-6):
        super().__init__()

        assert eps > 0, "eps must be greater than 0"
        assert 0 < alpha < 1, "alpha must be in the range (0, 1)"
        assert alpha >= eps, "alpha must be greater than or equal to eps"

        self.alpha = alpha
        self.eps = eps

    def logit(self, x: torch.Tensor) -> torch.Tensor:
        """Logit function with a small offset for numerical stability."""
        return torch.log(x / (1 - x + self.eps))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pre_logit = self.alpha + (1 - self.alpha) * x
        return self.logit(pre_logit)


class SigmoidTransform:
    """
    This is the inverse of the Logit transform above and has the following form:

    sigmoid(x) = 1 / (1 + exp(-x_tilde))

    where x_tilde = alpha + (1 - alpha) * x
    """

    def __init__(self, alpha=0.05):
        super().__init__()

        assert 0 < alpha < 1, "alpha must be in the range (0, 1)"

        self.alpha = alpha

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # get all the non-batch dimensions

        pre_sigmoid = torch.sigmoid(x)

        return (pre_sigmoid - self.alpha) / (1 - self.alpha)
