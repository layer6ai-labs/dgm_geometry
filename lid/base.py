import functools
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import torch

from data.transforms.unpack import UnpackBatch
from models.diffusions.sdes.utils import filter_kwargs_for_function

# Different types of input data that can be given to the LID estimator
LIDInputType = torch.Tensor | np.ndarray


def _infer_dim(x: torch.Tensor):
    # returns the shape of the tensor estimate.
    return x.numel() // x.shape[0]


class LIDEstimator(ABC):
    """
    Abstraction of LID estimation methods including both model-based and
    model-free LID estimators.
    """

    @abstractmethod
    class Artifact(ABC):
        """
        This class represents an artifact that is used to estimate the LID of a point.
        It is typically used for preprocessing purposes. For example, if your LID estimation
        involves taking a Jacobian of the datapoint and then computing the LID as a function
        of that Jacobian, you may want to store the Jacobian as an artifact. This way,
        when you are re-running the LID estimation, you can skip the Jacobian computation.
        """

    def __init__(
        self,
        data,
        ambient_dim: int | None = None,
        device: torch.device | None = None,
        unpack: UnpackBatch | None = None,
    ) -> None:
        """
        Initialize the estimator with the data that we are planning to use for LID estimation
        and also take in an optional ground truth LID if it is available
        """
        self.data = data
        self.ambient_dim = ambient_dim
        self.device = device
        if unpack is None:
            self.unpack = UnpackBatch()
        else:
            self.unpack = unpack

    @abstractmethod
    def fit(
        self,
    ):
        """
        Fit the estimator to the data and do one-time processing on the model if necessary
        before using the lid estimation methods.
        """

    def _preprocess(
        self,
        x: LIDInputType | Iterable[LIDInputType],
        **kwargs,
    ) -> Artifact:
        """
        The actual function that should be implemented by the child class to preprocess the data
        """
        raise NotImplementedError("The _preprocess method should be implemented.")

    @functools.wraps(_preprocess)
    def preprocess(
        self,
        x: LIDInputType | Iterable[LIDInputType],
        **kwargs,
    ) -> Artifact:
        """
        Store data and perform any preprocessing necessary for LID estimation on that
        particular set of data. This is useful for caching and speedup purposes.

        Args:
            x: A batch [batch_size, data_dim] or an iterable over the batches of data points at which to estimate the LID.
        """
        x = self.unpack(x)
        inferred = _infer_dim(x)
        if self.ambient_dim is None:
            self.ambient_dim = inferred
        else:
            assert (
                self.ambient_dim == inferred
            ), f"Multiple data points with different dimensions are not allowed! previously got {self.ambient_dim} and now got {inferred}"
        return self._preprocess(x, **kwargs)

    def compute_lid_from_artifact(
        self,
        lid_artifact: Artifact | None = None,
        **kwargs,
    ):
        """
        Compute the LID for the buffered data, but with different settings
        that are specified in the kwargs. This is useful for caching and speedup purposes, because many times
        we keep the data the same but change the scale.

        Args:
            scale (Optional[Union[float, int]]):
                The scale at which to estimate the LID. If None, the scale will be estimated from the data
                when set to None, the scale will be set automatically.
        Returns:
            lid: A batch [batch_size, data_dim] or an iterable over the batches of LID estimates, depending on the buffer type.
        """

    def _estimate_lid(
        self,
        x: LIDInputType | Iterable[LIDInputType],
        **kwargs,
    ):
        """
        Estimate the local intrinsic dimension of the data at given points.
        The input is batched, so the output should be batched as well.

        One can also set a number of parameters using the kwargs to
        customize the LID estimation process. For example,
        one might want to set a threshold on the Jacobian singular values, here,
        one can set that threshold.

        Args:
            x:
                A batch [batch_size, data_dim] or an iterable over the batches of data points at which to estimate the LID.
        Returns:
            lid:
                Returns a batch (batch_size, ) or iterable of LID values for the input data, depending on the input type.
        """
        kwargs_filterd = filter_kwargs_for_function(self._preprocess, x=x, **kwargs)
        artifact = self._preprocess(**kwargs_filterd)
        # check if the type of the artifact can be found within the scope of the class that 'x' belongs to
        class_of_self = self.__class__
        assert hasattr(
            class_of_self, "Artifact"
        ), "The class of the input data should have an attribute `Artifact`"
        # check if in the namespace class_of_self, `Artifact` is defined
        artifact_class = getattr(class_of_self, "Artifact")
        assert isinstance(
            artifact, artifact_class
        ), f"The artifact should be of type {artifact_class}"

        kwargs_filterd = filter_kwargs_for_function(
            self.compute_lid_from_artifact, lid_artifact=artifact, **kwargs
        )
        return self.compute_lid_from_artifact(**kwargs_filterd)

    @functools.wraps(_estimate_lid)
    def estimate_lid(
        self,
        x: LIDInputType | Iterable[LIDInputType],
        **kwargs,
    ):
        x = self.unpack(x)
        inferred = _infer_dim(x)
        if self.ambient_dim is None:
            self.ambient_dim = inferred
        else:
            assert (
                self.ambient_dim == inferred
            ), f"Multiple data points with different dimensions are not allowed! previously got {self.ambient_dim} and now got {inferred}"
        return self._estimate_lid(x, **kwargs)


# TODO: update documentation once finalized!
class ModelBasedLIDEstimator(LIDEstimator):
    """
    An abstract class for estimators that use a generative model implemented in torch
    (as a torch.nn.Module). One can either pretrain the model and pass it to the estimator
    or pass a training function alongside the data for the LID estimator to train the model
    in the `fit` method.

    Two examples of such a method are LIDL (https://arxiv.org/abs/2206.14882), and the
    model-based LID estimators discussed here: https://arxiv.org/abs/2403.18910.
    """

    class Artifact:
        pass

    def __init__(
        self,
        model: torch.nn.Module,
        ambient_dim: int | None = None,
        device: torch.device | None = None,
        unpack: UnpackBatch | None = None,
    ):
        """
        Args:
            ambient_dim:
                The dimension of the data. If not specified, it will be inferred from the first data point.
            model:
                A torch Module that is a likelihood-based deep generative model and that one can
                compute LID from.
            data:
                The data that the model will be trained/evaluated on for LID.
            device:
                The device on which the model will be trained.
            unpack:
                An unpacking function that will be used to unpack the data. If not specified, the default
                unpacking function does nothing. Unpacking simply takes a batch of the dataset and returns
                the actual content of the batch.
        """

        super().__init__(
            data=None,
            ambient_dim=ambient_dim,
            device=device,
            unpack=unpack,
        )
        # instead of raising ValueError, do an assert
        assert isinstance(model, torch.nn.Module), "The model should be a torch Module"

        self.model = model
        # if no train function is specified, the train function will only return
        # the model untouched!
        # if model has at least a parameter
        model_device = None
        if len(list(self.model.parameters())) > 0:
            model_device = next(self.model.parameters()).device

        self.device = device if device is not None else model_device
        assert self.device is not None, "The device should be specified"
        self.model = self.model.to(self.device)

    def fit(
        self,
        **training_kwargs,
    ):
        """Fit the estimator to the data"""
        assert (
            False
        ), "For model-based LID, you cannot call fit, it is assumed that the model is already pretrained!"
