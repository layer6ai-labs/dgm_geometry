from pathlib import Path
import os

import hydra
import lightning as L
import mlflow
import torch
import torchvision.transforms.functional as TVF
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, errors as omegaconf_errors
from pprint import pprint

# get the environment variable IS_TESTING
# when testing, the main function will be called from a
# different file, so we need to import the tools module
# via the scripts package
if os.environ.get("IS_TESTING", False):
    from scripts import tools
else:
    import tools


@hydra.main(version_base=None, config_path="../conf/", config_name="train")
@tools.MlflowDecorator(
    exclude_attributes=[  # The hydra attributes to remove from the mlflow logging
        "all_callbacks",
        "dgm",
        "all_data_transforms",
        "all_sampling_transforms",
        "lightning_dgm",
        "_all_callbacks",
        "_all_data_transforms",
        "_all_sampling_transforms",
    ],
    out_dir="./outputs",  # The directory where the artifacts are stored
    experiment_name="Training",  # The name of the experiment to be logged on mlflow
)
def main(cfg: DictConfig, artifact_dir: Path):

    # Configure dataset
    train_dataset = instantiate(cfg.dataset.train)

    split_size = cfg.dataset.val.get("split_size")
    if split_size is not None:
        if split_size == 0:  # No validation data
            val_loader = None
        elif split_size > 0:  # Split validation data out of the training data
            split = train_dataset.train_test_split(test_size=split_size)
            train_dataset = split["train"]
            val_dataset = split["test"]
            val_loader = torch.utils.data.DataLoader(val_dataset, **cfg.train.val_loader)
    else:  # cfg.dataset.val is a separate dataset object
        val_dataset = instantiate(cfg.dataset.val)
        val_loader = torch.utils.data.DataLoader(val_dataset, **cfg.train.val_loader)

    train_loader = torch.utils.data.DataLoader(train_dataset, **cfg.train.loader)

    # Configure model and training
    lightning_dgm = instantiate(cfg.train.lightning_dgm)  # instantiate the LightningDGM model
    # Train
    trainer = L.Trainer(default_root_dir=artifact_dir, **instantiate(cfg.train.trainer))

    trainer.fit(
        model=lightning_dgm,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.train.ckpt_path,
    )


if __name__ == "__main__":

    tools.setup_root()
    main()
