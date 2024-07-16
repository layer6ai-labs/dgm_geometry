from torch.utils.data import Dataset as TorchDataset


class HuggingFaceDataset:
    """Combine a datasets.arrow_dataset.Dataset and a transform into a single class"""

    def __init__(self, dataset, subset_size=None, class_filter=None, transform=None):
        self.dataset = dataset

        if class_filter is not None:
            self.dataset = dataset.filter(lambda row: row["label"] in class_filter)

        if transform is not None:
            self.transform = self.get_hf_transform(transform)
            self.dataset.set_transform(self.transform)
        else:
            self.transform = None

        if subset_size is not None:
            split = self.dataset.train_test_split(test_size=1, train_size=subset_size)
            self.dataset = split["train"]

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def get_hf_transform(transform):
        """Turn a torchvision-style transform into one for a huggingface dataset"""

        def hf_transform(datarow):
            # A datarow (from a huggingface dataset) is dict-like and sometimes has
            # different names for the image column - try a couple
            img_keys = {"img", "image"}
            for img_key in img_keys:
                if img_key in datarow:
                    image_pils = datarow[img_key]

            # Apply preprocessing transform  specified in config to the images
            image_tensors = [transform(image.convert("RGB")) for image in image_pils]
            return {"images": image_tensors}

        return hf_transform


class HuggingFaceSubset(HuggingFaceDataset):
    def __init__(self, dataset, size, transform=None):
        super().__init__(dataset, transform=None)
        self.dataset = self.dataset.train_test_split(test_size=0, train_size=size)["train"]


class TorchHuggingFaceDatasetWrapper(TorchDataset):
    """
    Turn the huggingface dataset into a torch dataset that only returns the images
    in the huggingface dataset. This is useful to pass on to other methods that
    explicitly require a torch dataset. such as LID estimation.
    """

    def __init__(
        self,
        hugging_face_dataset: HuggingFaceDataset,
    ):
        # perform the subsampling using the train_test_split function
        self.hugging_face_dataset = hugging_face_dataset

    def __len__(self):
        return len(self.hugging_face_dataset)

    def __getitem__(self, idx):
        return self.hugging_face_dataset.dataset[idx]["images"]
