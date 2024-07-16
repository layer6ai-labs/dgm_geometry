"""
This test is used primarily to check if the model design is
backwards compatible and that one can use the checkpoint URL
that is specified in the downloads directory.
"""

import os
from datetime import datetime
from pathlib import Path

import pytest

from scripts.download_resources import main as download_checkpoints_main
from scripts.train import main as train_main

from .utils import hydra_script_runner


@pytest.fixture
def default_overrides():
    """Returns a list of overrides that should happen for all of the configurations specified here"""
    current_datatime = datetime.now().strftime("%H:%M:%S_of_%y-%m-%d")  # add a timestamp
    return [
        "++mflow.experiment_name=hydra_test",
        "+mlflow.tags.timestamp=" + current_datatime,
        "+mlflow.tags.test_script=train",
        "+mlflow.tags.test_type=checkpoints",
    ]


@pytest.fixture
def ground_truth_yaml_directory() -> Path:
    """Contains the directory containing the reference yaml files"""
    directory = Path(__file__).parent.parent / "resources" / "hydra_config" / "model_checkpoints"
    os.makedirs(directory, exist_ok=True)
    return directory


@pytest.fixture
def generated_yaml_directory() -> Path:
    """All the generated yaml files will be dumped into this directory and then compared to the ground truth ones."""
    directory = (
        Path(__file__).parent.parent.parent / "outputs" / "hydra_config" / "model_checkpoints"
    )
    os.makedirs(directory, exist_ok=True)
    return directory


# diffusion models for image data
fmnist_diffusion_unet = [
    "dataset=fmnist",
    "+experiment=train_diffusion_greyscale",
    "+checkpoint=diffusion_fmnist",
]
mnist_diffusion_unet = [
    "dataset=mnist",
    "+experiment=train_diffusion_greyscale",
    "+checkpoint=diffusion_mnist",
]
svhn_diffusion_unet = [
    "dataset=svhn",
    "+experiment=train_diffusion_rgb",
    "+checkpoint=diffusion_svhn",
]
cifar10_diffusion_unet = [
    "dataset=cifar10",
    "+experiment=train_diffusion_rgb",
    "+checkpoint=diffusion_cifar10",
]
# diffusion models for image data but with MLPs instead
fmnist_diffusion_mlp = [
    "dataset=fmnist",
    "+experiment=train_diffusion_greyscale_flattened",
    "+checkpoint=diffusion_fmnist_mlp",
]
mnist_diffusion_mlp = [
    "dataset=mnist",
    "+experiment=train_diffusion_greyscale_flattened",
    "+checkpoint=diffusion_mnist_mlp",
]
svhn_diffusion_mlp = [
    "dataset=svhn",
    "+experiment=train_diffusion_rgb_flattened",
    "+checkpoint=diffusion_svhn_mlp",
]
cifar10_diffusion_mlp = [
    "dataset=cifar10",
    "+experiment=train_diffusion_rgb_flattened",
    "+checkpoint=diffusion_cifar10_mlp",
]
# flow models
mnist_flow = [
    "dataset=mnist",
    "+experiment=train_flow_greyscale",
    "+checkpoint=flow_mnist",
]
fmnist_flow = [
    "dataset=fmnist",
    "+experiment=train_flow_greyscale",
    "+checkpoint=flow_fmnist",
]


# A list of all possible settings to test:
all_settings = [
    (0, "dev_fmnist_diffusion_unet", fmnist_diffusion_unet),
    (1, "dev_mnist_diffusion_unet", mnist_diffusion_unet),
    (2, "dev_svhn_diffusion_unet", svhn_diffusion_unet),
    (3, "dev_cifar10_diffusion_unet", cifar10_diffusion_unet),
    (4, "dev_fmnist_diffusion_mlp", fmnist_diffusion_mlp),
    (5, "dev_mnist_diffusion_mlp", mnist_diffusion_mlp),
    (6, "dev_svhn_diffusion_mlp", svhn_diffusion_mlp),
    (7, "dev_cifar10_diffusion_mlp", cifar10_diffusion_mlp),
    (8, "dev_mnist_flow", mnist_flow),
    (9, "dev_fmnist_flow", fmnist_flow),
]


@pytest.mark.parametrize(
    "setting",
    all_settings,
)
@pytest.mark.parametrize(
    "dummy",
    [True, False],
)
def test_checkpoints_scripts(
    default_overrides,
    setting,
    ground_truth_yaml_directory,
    generated_yaml_directory,
    dummy,
):
    level, setting_name, new_overrides = setting
    overrides = (default_overrides or []) + (new_overrides or [])
    overrides += ["++mlflow.tags.setting=" + setting_name]
    overrides += ["++mlflow.experiment_name=hydra_tests"]

    # run the setup only for one setting, this is to check that the
    # initilization step of the repository works with the given configurations.
    run_setup = os.getenv("RUN_SETUP", default="False") == "True"
    if run_setup and level == 0 and dummy:
        download_checkpoints_main(download_files=True)

    hydra_script_runner(
        script_level=level,
        setting_name=setting_name,
        overrides=overrides,
        ground_truth_yaml_directory=ground_truth_yaml_directory,
        generated_yaml_directory=generated_yaml_directory,
        dummy=dummy,
        main_fn=train_main,
        script_name="train",
        exclude_attributes=[
            "mlflow.tags.timestamp",  # exclude the timestamp from the mlflow tags when comparing configurations
            "train.ckpt_path",  # this is a private info
        ],
    )
