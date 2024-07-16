"""
Use likelihoods as a metric to evaluate models
"""

from .datapoint_metric import DatapointMetric


class LikelihoodMetric(DatapointMetric):

    def __init__(self, model, device=None, **likelihood_kwargs):
        assert hasattr(
            model, "log_prob"
        ), f"The model should be a likelihood-based DGM, but {model} does not have a log_prob method"
        super().__init__(model=model, device=device)
        self.likelihood_kwargs = likelihood_kwargs

    def score_batch(self, batch):
        return self.model.log_prob(batch, **self.likelihood_kwargs)
