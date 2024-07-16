from collections.abc import Collection
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from ..transforms.unpack import UnpackBatch


class EncodedDataset(TorchDataset):
    """A dataset constructed by applying an encoder nn.Module to another dataset."""

    def __init__(
        self,
        encoder: nn.Module,
        dataset: Collection,
        encode_on_init: bool = True,
        device: torch.device | None = None,
        batch_size: int = 1,
        unpack: UnpackBatch = None,
        verbose: bool = True,
    ):
        """
        Args:
            encoder: The nn.Module used to encode the dataset.
            dataset: The original dataset.
            encode_on_init: A flag indicating whether to encode the dataset in bulk on init, or to
                encode on the fly.
            device: The device on which to perform encoding. If encoding occurs on init, the
                encoder will be moved back to its original device afterwards.
            batch_size: (encode_on_init only) batch size for the encoding process.
            unpack: Callable used to convert batches from dataset into input for encoder.
            verbose: (encode_on_init only) enables a progress bar for the encoding process.
        """
        self.encoder = encoder
        self.raw_dataset = dataset
        self.encoded_data = {}
        if unpack is None:
            self.unpack = lambda x: x
        else:
            self.unpack = unpack

        if encode_on_init:
            self._encode_dataset(batch_size, device, verbose=verbose)
        else:
            self.encoder.to(device)

    @torch.no_grad()
    def _encode_dataset(self, batch_size, inference_device=None, verbose=True):
        encoder_device = next(self.encoder.parameters()).device
        if inference_device is not None:
            self.encoder.to(inference_device)

        num_batches = (len(self.raw_dataset) + batch_size - 1) // batch_size
        if verbose:
            iterable = tqdm(range(num_batches), desc="Encoding dataset")
        else:
            iterable = range(num_batches)

        for batch_idx in iterable:
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(self.raw_dataset))
            batch = torch.stack(
                [self.unpack(self.raw_dataset[i]) for i in range(batch_start, batch_end)]
            )
            encoded_batch = self.encoder(batch.to(inference_device)).cpu()
            for idx, encoded_datum in enumerate(encoded_batch):
                self.encoded_data[batch_start + idx] = encoded_datum

        self.encoder.to(encoder_device)

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        if index not in self.encoded_data:
            datum = self.raw_dataset[index]
            datum = self.unpack(datum)
            self.encoded_data[index] = self.encoder(datum[None, :])
        return self.encoded_data[index]


class EncodedTextImageDataset(TorchDataset):
    """Encodes a dataset of text-image pairs.

    Expects a dataset with the following properties:
    1. __getitem__ returns tuples with image in the first index and text in the second.
    2. Dataset has as property a `metadata` dictionary containing lists `image_path` and `caption`

    If you pass in an `encodings_path` pickle location, this dataset will automatically save
    encodings to and load them from that location.

    TODO: Create an abstract class to document this type of dataset
    """

    @torch.no_grad()
    def __init__(
        self,
        image_encoder,
        prompt_encoder,
        raw_dataset,
        image_batch_size=32,
        prompt_batch_size=128,
        encodings_path=None,
        **encoding_kwargs,
    ):
        """
        Args:
            image_encoder: The nn.Module used to encode the images.
            prompt_encoder: The nn.Module used to encode the text prompts.
            raw_dataset: The original dataset containing image-text pairs. See class-level
                docstring for details.
            image_batch_size: Batch size for the image encoding process.
            prompt_batch_size: Batch size for the prompt encoding process.
            encodings_path: Path to save/load encodings from disk.
            **encoding_kwargs: Additional keyword arguments for the underlying
                EncodedDataset containing the encoded images.
        """
        # Get metadata from underlying dataset
        self.metadata = raw_dataset.metadata

        if encodings_path:
            encodings_path = Path(encodings_path)

        # Get encoded data
        if encodings_path and encodings_path.exists():  # Load encodings from disk
            encoding_dict = torch.load(encodings_path)
            self.image_encodings = encoding_dict["image_encodings"]
            self.prompt_encodings = encoding_dict["prompt_encodings"]
        else:  # Generate encodings on the fly
            # Image encodings
            image_encodings = EncodedDataset(
                encoder=image_encoder,
                dataset=raw_dataset,
                encode_on_init=True,
                batch_size=image_batch_size,
                unpack=UnpackBatch([0]),
                **encoding_kwargs,
            )
            self.image_encodings = torch.stack(tuple(image_encodings.encoded_data.values()))

            # Prompt encodings
            self.prompt_encodings = []
            prompt_iterable = tqdm(  # "Batchify" list of prompts
                (
                    self.metadata["caption"][i : i + prompt_batch_size].tolist()
                    for i in range(0, len(self.metadata["caption"]), prompt_batch_size)
                ),
                desc="Encoding prompts",
            )
            for prompt in prompt_iterable:
                self.prompt_encodings.append(prompt_encoder(prompt).cpu())
            self.prompt_encodings = torch.cat(self.prompt_encodings)

            if encodings_path:  # Save encodings to disk
                encoding_dict = {
                    "image_encodings": self.image_encodings,
                    "prompt_encodings": self.prompt_encodings,
                }
                torch.save(encoding_dict, encodings_path)

    def __getitem__(self, idx):
        return self.image_encodings[idx], self.prompt_encodings[idx]

    def __len__(self):
        return len(self.image_encodings)
