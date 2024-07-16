import torch
import torch.distributions as dist


class VonMisesEuclidean(dist.Distribution):
    def __init__(self, loc=0.0, concentration=2.0, radius=1.0, centre=(0, 0)):
        loc = torch.Tensor([loc])
        concentration = torch.Tensor([concentration])
        self.radius = torch.Tensor([radius])
        self.centre = torch.Tensor(centre)
        self.gt = dist.VonMises(loc, concentration)

    def _transform(self, thetas):
        return self._polar_to_euclidean(thetas) * self.radius + self.centre

    def sample(self, sample_shape):
        thetas = self.gt.sample(sample_shape)
        data = self._transform(thetas)
        return data

    @staticmethod
    def _polar_to_euclidean(theta):
        return torch.cat((torch.cos(theta), torch.sin(theta)), dim=-1)

    @staticmethod
    def _euclidean_to_polar(xy):
        return torch.arctan2(xy[:, 1], xy[:, 0])[:, None]
