"""
This code contains processing on batches to make them ready for training.

For example, if a dataloader is set on an LID dataset, it will contain batches of form:
"x, lid, idx"
the unpack batch function takes in this tuple and spits out the x value only.

As another example, HuggingFace datasets are wrapped in dictionaries. An example batch
from an image dataset in hugging face will look like:
{"images": tensor, ...}
or
{"img": tensor, ...}
the unpack function will extract the actual image tensor from the dictionary, making
it ready to be passed on to a loss function.
"""

from typing import Any, List

import torch


class UnpackBatch:
    """
    A generic unpacker that follows a set of access tokens to unpack.

    As an example, consider a batch following the scheme below:
    {
        "x": {
            "y": tuple(tensor_content, ...)
        }
    }
    then the access_tokens will be ["x", "y", 0] to access the actual content of the batch.
    """

    def __init__(
        self,
        access_tokens: List[str | int] | None = None,
    ):
        """
        Args:
            access_tokens (List[str | int], optional):
                A list of access tokens that are used to unpack the batch.
                For example, if the batch is a dictionary, the access tokens
                will be used to access the underlying data. Defaults to None
                which means no unpacking is done.
        """
        self.access_tokens = access_tokens or []

    def __call__(self, batch: Any) -> torch.Tensor:
        if len(self.access_tokens) == 0:
            return batch

        ret_batch = batch
        # Fetch the item from the base dataset
        current_path = []
        for key in self.access_tokens:
            assert not isinstance(
                ret_batch, torch.Tensor
            ), f"{current_path}: Reached a tensor before unpacking!"
            if "__getitem__" in dir(ret_batch):
                ret_batch = ret_batch[key]
            else:
                assert hasattr(
                    ret_batch, key
                ), f"{current_path}: Could not access '{key}' from {type(ret_batch)}"
                ret_batch = getattr(ret_batch, key)
            current_path = current_path + [key]
        assert isinstance(
            ret_batch, torch.Tensor
        ), f"Expected a tensor after unpacking, but got {type(ret_batch)}"
        return ret_batch


class UnpackHuggingFace(UnpackBatch):
    def __init__(self):
        pass

    def __call__(self, batch):
        """Some batches are wrapped in dictionaries or tuples; unpack the underlying datapoint"""

        if isinstance(batch, torch.Tensor):
            return batch

        elif isinstance(batch, (tuple, list)):
            return batch[0]

        else:
            batch_keys = {"datapoint", "img", "images"}
            for batch_key in batch_keys:
                if batch_key in batch:
                    return batch[batch_key]

        raise ValueError("Could not unpack batch from dataloader")


class UnpackTabular(UnpackBatch):
    # It can unpack both LID synthetic data and the tabular data without any additional arguments
    def __call__(self, batch):
        """Some batches are wrapped in dictionaries or tuples; unpack the underlying datapoint"""
        if torch.is_tensor(batch):
            return batch
        return batch[0]
