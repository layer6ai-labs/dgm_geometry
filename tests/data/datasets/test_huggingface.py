import datasets
import torchvision
from torch.utils.data import Dataset as TorchDataset

from data.datasets import HuggingFaceDataset, TorchHuggingFaceDatasetWrapper


def test_cifar10_hugging_face():
    cifar10_arrow_dataset = datasets.load_dataset(
        path="cifar10", split="test", trust_remote_code=True
    )
    assert isinstance(cifar10_arrow_dataset, datasets.Dataset)
    our_dset = HuggingFaceDataset(
        dataset=cifar10_arrow_dataset, subset_size=100, transform=torchvision.transforms.ToTensor()
    )
    assert len(our_dset) == 100
    wrapped_dset = TorchHuggingFaceDatasetWrapper(our_dset)
    assert isinstance(wrapped_dset, TorchDataset)
    assert len(wrapped_dset) == 100
    assert (
        wrapped_dset[0].shape[0] == 3
        and wrapped_dset[0].shape[1] == 32
        and wrapped_dset[0].shape[2] == 32
    )
