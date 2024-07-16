import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset

from .hugging_face import HuggingFaceDataset


class Tuxemon(TorchDataset):
    def __init__(self, subset_size=None, transform=None, seed=42, timeout=5):
        self.dataset = HuggingFaceDataset(
            load_dataset("diffusers/tuxemon", split="train", trust_remote_code=True),
            subset_size=subset_size,
            datapoint_name="image",
            label_name="gpt4_turbo_caption",
        )
        self.transform = transform
        self.metadata = pd.DataFrame(
            {
                "caption": self.dataset["label"],
            }
        )

    def __getitem__(self, idx):
        row = self.dataset[idx]
        img, label = row["datapoint"], row["label"]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)
