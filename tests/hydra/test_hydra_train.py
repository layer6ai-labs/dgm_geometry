"""
Script for testing all the possible configurations we support 
by running the 

python scripts/train.py ...

all of these configurations are stored and they are boiled down to a lightweight
version so that the runtime does not take too long.
"""

import os
from datetime import datetime
from pathlib import Path

import pytest

from scripts.train import main as train_main

from .utils import hydra_script_runner


@pytest.fixture
def default_overrides():
    """Returns a list of overrides that should happen for all of the configurations specified here"""
    current_datatime = datetime.now().strftime("%H:%M:%S_of_%y-%m-%d")
    return [
        "++mlflow.experiment_name=hydra_test",
        "+mlflow.tags.timestamp=" + current_datatime,
        "+mlflow.tags.test_script=train",
        "train.loader.batch_size=10",  # make the batch size smaller to speed up the tests
        "train.val_loader.batch_size=10",
    ]


@pytest.fixture
def ground_truth_yaml_directory() -> Path:
    """Contains the directory containing the reference yaml files"""
    directory = Path(__file__).parent.parent / "resources" / "hydra_config" / "train"
    os.makedirs(directory, exist_ok=True)
    return directory


@pytest.fixture
def generated_yaml_directory() -> Path:
    """All the generated yaml files will be dumped into this directory and then compared to the ground truth ones."""
    directory = Path(__file__).parent.parent.parent / "outputs" / "hydra_config" / "train"
    os.makedirs(directory, exist_ok=True)
    return directory


# A list of all possible overrides being used
cifar10_diffusion = [
    "dataset=cifar10",
    "+experiment=train_diffusion_rgb",
    "train.trainer.fast_dev_run=True",
    "+train.trainer.devices=1",  # set it to 1 for efficiency
]
mnist_diffusion = [
    "dataset=mnist",
    "+experiment=train_diffusion_greyscale",
    "train.trainer.fast_dev_run=True",
    "+train.trainer.devices=1",  # set it to 1 for efficiency
]
svhn_flow = [
    "dataset=svhn",
    "+experiment=train_flow_rgb",
    "train.trainer.fast_dev_run=True",
    "+train.trainer.devices=1",  # set it to 1 for efficiency
]
fmnist_flow = [
    "dataset=fmnist",
    "+experiment=train_flow_greyscale",
    "train.trainer.fast_dev_run=True",
    "+train.trainer.devices=1",  # set it to 1 for efficiency
]
tabular_diffusion = [
    "dataset=lollipop",
    "+experiment=train_diffusion_tabular",
    "train.trainer.fast_dev_run=True",
    "+train.trainer.devices=1",  # set it to 1 for efficiency
]

tabular_diffusion_with_umap = [
    "dataset=lollipop",
    "+experiment=train_diffusion_tabular",
    "train.trainer.fast_dev_run=True",
    "+train.trainer.devices=1",  # set it to 1 for efficiency
    "+callbacks@all_callbacks.umap=umap",
    "all_callbacks.umap.frequency=1",
]

cifar10_diffusion_with_umap = [
    "dataset=cifar10",
    "+experiment=train_diffusion_rgb",
    "train.trainer.fast_dev_run=True",
    "+train.trainer.devices=1",  # set it to 1 for efficiency
    "+callbacks@all_callbacks.umap=umap",
    "all_callbacks.umap.frequency=1",
]


tabular_flow_training = [
    "dataset=lollipop",
    "+experiment=train_flow_tabular",
    "dataset.train.size=4096",
    "train.trainer.fast_dev_run=True",
    "+train.trainer.devices=1",  # set it to 1 for efficiency
]

cifar10_diffusion_with_umap_callback = [
    "dataset=cifar10",
    "+experiment=train_diffusion_rgb",
    "train.trainer.fast_dev_run=True",
    "+callbacks@all_callbacks.umap=umap",
    "all_callbacks.umap.frequency=1",
    "all_callbacks.umap.limit_count=10",
    "+train.trainer.devices=1",  # set it to 1 for efficiency
]

mnist_flow_mix_and_match = [
    "dataset=mnist",
    "+experiment=train_flow_greyscale",
    "train.trainer.fast_dev_run=True",
    "+train.trainer.devices=1",  # set it to 1 for efficiency
    "+callbacks@all_callbacks.umap=umap",
    "all_callbacks.umap.frequency=1",
]
flow_without_logit_transform = [
    "dataset=mnist",
    "+experiment=train_flow_greyscale",
    "data_transforms@all_data_transforms.t4=identity",  # remove the logit transform
    "data_transforms@all_sampling_transforms.t0=identity",  # remove the sigmoid transform
    "data_transforms@all_data_transforms.t0=identity",
    "train.trainer.fast_dev_run=True",
    "+train.trainer.devices=1",  # set it to 1 for efficiency
]

normal_bundles_lollipop = [
    "dataset=lollipop",
    "+experiment=train_diffusion_tabular",
    "train.trainer.fast_dev_run=True",
    "+train.trainer.devices=1",  # set it to 1 for efficiency
    "+callbacks@all_callbacks.lid=normal_bundle_lid_curve",
    "all_callbacks.lid.frequency=1",
    "all_callbacks.lid.subsample_size=128",
    "all_callbacks.lid.lid_preprocessing_args.noise_time=0.1",
    "all_callbacks.lid.lid_preprocessing_args.num_scores=null",
    "all_callbacks.lid.lid_preprocessing_args.score_batch_size=128",
    "all_callbacks.lid.batch_size=8",
]

normal_bundles_cifar10 = [
    "dataset=cifar10",
    "+experiment=train_diffusion_rgb",
    "train.trainer.fast_dev_run=True",
    "+train.trainer.devices=1",  # set it to 1 for efficiency
    "+callbacks@all_callbacks.lid=normal_bundle_lid_curve",
    "all_callbacks.lid.frequency=1",
    "all_callbacks.lid.subsample_size=128",
    "all_callbacks.lid.lid_preprocessing_args.noise_time=0.1",
    "all_callbacks.lid.lid_preprocessing_args.num_scores=10",
    "all_callbacks.lid.lid_preprocessing_args.score_batch_size=128",
    "all_callbacks.lid.batch_size=8",
]

lidl_lollipop = [
    "dataset=lollipop",
    "+experiment=train_lidl_tabular",
    "dataset.train.size=4096",
    "dataset.val.size=128",
    "train.trainer.fast_dev_run=True",
    "+train.trainer.devices=1",  # set it to 1 for efficiency
    "+callbacks@all_callbacks.umap=umap",
]

# A list of all possible settings to test:
# The first element of the tuple is the index or script level of the test, typically higher level tests take longer
# The second element is the name of the setting that will be logged to mlflow and the name of the yaml file
# The third element is the list of overrides that will be passed to the script
all_settings = [
    (0, "dev_cifar10_diffusion", cifar10_diffusion),
    (1, "dev_mnist_diffusion", mnist_diffusion),
    (2, "dev_svhn_flow", svhn_flow),
    (3, "dev_fmnist_flow", fmnist_flow),
    (4, "dev_tabular_diffusion", tabular_diffusion),
    (5, "dev_mnist_flow_mix_and_match", mnist_flow_mix_and_match),
    (6, "dev_flow_without_logit_transform", flow_without_logit_transform),
    (7, "dev_tabular_flow_training", tabular_flow_training),
    (
        8,
        "dev_cifar10_diffusion_with_umap_callback",
        cifar10_diffusion_with_umap_callback,
    ),
    (9, "dev_normal_bundles_lollipop", normal_bundles_lollipop),
    (10, "dev_tabular_diffusion_with_umap", tabular_diffusion_with_umap),
    (11, "dev_cifar10_diffusion_with_umap", cifar10_diffusion_with_umap),
    (12, "dev_normal_bundles_cifar10", normal_bundles_cifar10),
    (13, "dev_lidl_lollipop", lidl_lollipop),
]


@pytest.mark.parametrize(
    "setting",
    all_settings,
)
@pytest.mark.parametrize(
    "dummy",
    [True, False],
)
def test_train_scripts(
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
        main_fn=train_main,
        script_name="train",
        exclude_attributes=[
            "mlflow.tags.timestamp",  # exclude the timestamp from the mlflow tags when comparing configurations
        ],
    )
