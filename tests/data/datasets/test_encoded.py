import torch
import torch.nn as nn

from data.datasets import EncodedDataset


def test_encode_on_the_fly():
    class TimesTwoEncoder(nn.Module):
        def forward(self, x):
            return 2 * x

    encoded_dataset = EncodedDataset(
        encoder=TimesTwoEncoder(),
        dataset=torch.ones((8, 3)),
        encode_on_init=False,
    )

    assert torch.eq(encoded_dataset[0], torch.tensor([2.0, 2.0, 2.0])).all()
