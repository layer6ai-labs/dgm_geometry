from .datapoint_metric import DatapointMetric
from lid.base import ModelBasedLIDEstimator

"""A simple LID detector"""


class SimpleLIDMetric(DatapointMetric):
    def __init__(self, model, lid_estimator_partial, **estimate_lid_kwargs):
        super().__init__(model=model)
        self.lid_estimator: ModelBasedLIDEstimator = lid_estimator_partial(model)
        self.estimate_lid_kwargs = estimate_lid_kwargs

    def score_batch(self, batch):
        return self.lid_estimator.estimate_lid(batch, **self.estimate_lid_kwargs)
