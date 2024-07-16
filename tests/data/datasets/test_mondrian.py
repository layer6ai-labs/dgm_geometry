from data.datasets import MondrianDataset


def test_mondrian():
    size = 5

    dataset = MondrianDataset(size=size)
    assert len(dataset) == size
    assert dataset[0]["datapoint"].shape == (3, 32, 32)
