"""
Script for testing all the possible configurations we support 
by running the 

python scripts/model_free_lid.py ...

all of these configurations are stored and they are boiled down to a lightweight
version so that the runtime does not take too long.
"""

import os
from datetime import datetime
from pathlib import Path

import pytest

from scripts.model_free_lid import main as model_free_lid_main

from .utils import hydra_script_runner


@pytest.fixture
def default_overrides():
    """Returns a list of overrides that should happen for all of the configurations specified here"""
    current_datatime = datetime.now().strftime("%H:%M:%S_of_%y-%m-%d")  # add a timestamp
    return [
        "++mflow.experiment_name=hydra_test",
        "+mlflow.tags.timestamp=" + current_datatime,
        "+mlflow.tags.test_script=model_free_lid",
        "subsample_size=1024",
    ]


@pytest.fixture
def ground_truth_yaml_directory() -> Path:
    """Contains the directory containing the reference yaml files"""
    directory = Path(__file__).parent.parent / "resources" / "hydra_config" / "model_free_lid"
    os.makedirs(directory, exist_ok=True)
    return directory


@pytest.fixture
def generated_yaml_directory() -> Path:
    """All the generated yaml files will be dumped into this directory and then compared to the ground truth ones."""
    directory = Path(__file__).parent.parent.parent / "outputs" / "hydra_config" / "model_free_lid"
    os.makedirs(directory, exist_ok=True)
    return directory


# all the different different command overrides:
lollipop_cfdm = ["dataset=lollipop", "lid_method=cfdm", "+experiment=lid_tabular"]
lollipop_ess = ["dataset=lollipop", "lid_method=ess", "+experiment=lid_tabular"]
lollipop_lpca = [
    "dataset=lollipop",
    "lid_method=lpca",
    "+experiment=lid_tabular",
    "subsample_size=4096",
]
swiss_roll_ess = ["dataset=swiss_roll", "lid_method=ess", "+experiment=lid_tabular"]
affine_800D_200d_unifrom_cfdm = [
    "lid_method=cfdm",
    "+experiment=lid_tabular",
    "dataset=manifolds/large/affine_800D_200d_uniform",
]
affine_800D_200d_unifrom_ess = [
    "lid_method=ess",
    "+experiment=lid_tabular",
    "dataset=manifolds/large/affine_800D_200d_uniform",
]
affine_800D_200d_unifrom_lpca = [
    "lid_method=lpca",
    "+experiment=lid_tabular",
    "subsample_size=4096",
    "dataset=manifolds/large/affine_800D_200d_uniform",
]
mnist_cfdm = ["dataset=mnist", "lid_method=cfdm", "+experiment=lid_greyscale"]
cifar10_cfdm = ["dataset=cifar10", "lid_method=cfdm", "+experiment=lid_rgb"]
mnist_ess = ["dataset=mnist", "lid_method=ess", "+experiment=lid_greyscale"]
cifar10_ess = ["dataset=cifar10", "lid_method=ess", "+experiment=lid_rgb"]

# A list of all possible settings to test:
# The first element of the tuple is the index or script level of the test, typically higher level tests take longer
# The second element is the name of the setting that will be logged to mlflow and the name of the yaml file
# The third element is the list of overrides that will be passed to the script
all_settings = [
    (0, "dev_lollipop_cfdm", lollipop_cfdm),
    (1, "dev_lollipop_ess", lollipop_ess),
    (2, "dev_lollipop_lpca", lollipop_lpca),
    (3, "dev_swiss_roll_ess", swiss_roll_ess),
    (4, "dev_affine_800D_200d_unifrom_cfdm", affine_800D_200d_unifrom_cfdm),
    (5, "dev_affine_800D_200d_unifrom_ess", affine_800D_200d_unifrom_ess),
    (6, "dev_affine_800D_200d_unifrom_lpca", affine_800D_200d_unifrom_lpca),
    (7, "dev_mnist_cfdm", mnist_cfdm),
    (8, "dev_cifar10_cfdm", cifar10_cfdm),
    (9, "dev_mnist_ess", mnist_ess),
    (10, "dev_cifar10_ess", cifar10_ess),
]


@pytest.mark.parametrize(
    "setting",
    all_settings,
)
@pytest.mark.parametrize(
    "dummy",
    [True, False],
)
def test_lid_scripts(
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
    hydra_script_runner(
        script_level=level,
        setting_name=setting_name,
        overrides=overrides,
        ground_truth_yaml_directory=ground_truth_yaml_directory,
        generated_yaml_directory=generated_yaml_directory,
        dummy=dummy,
        main_fn=model_free_lid_main,
        script_name="model_free_lid",
        exclude_attributes=[
            "mlflow.tags.timestamp",  # exclude the timestamp from the mlflow tags when comparing configurations
        ],
    )
