import pandas as pd
import requests
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

from .hugging_face import HuggingFaceDataset


class COCO(TorchDataset):
    def __init__(self, subset_size=None, transform=None, seed=42, timeout=5):
        self.dataset = HuggingFaceDataset(
            load_dataset(
                "ChristophSchuhmann/MS_COCO_2017_URL_TEXT", split="train", trust_remote_code=True
            ),
            subset_size=subset_size,
            label_name="TEXT",
        )
        self.timeout = timeout
        self.transform = transform
        self.metadata = pd.DataFrame(
            {
                "image_path": self.dataset["URL"],
                "caption": self.dataset["label"],
            }
        )

    def __getitem__(self, idx):
        row = self.dataset[idx]
        url, label = row["URL"], row["label"]

        img_response = requests.get(url, stream=True, timeout=self.timeout).raw
        img = Image.open(img_response)
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)
