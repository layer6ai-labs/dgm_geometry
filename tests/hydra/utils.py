"""
This file contains the meat of the hydra tests.
"""

import os
from pathlib import Path
from typing import Callable, List

import dotenv
import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.core.singleton import Singleton
from omegaconf import OmegaConf

from scripts import tools


def parse_test_indices(txt: str):
    """
    Take in a one-line script and parse it to get the test indices

    Examples:

    0-2 -> [0, 1, 2]
    0,2 -> [0, 2]
    1 -> [1]
    0,2-4,5 -> [0, 2, 3, 4, 5]
    """
    if "," in txt:
        ret = []
        for x in txt.split(","):
            ret += parse_test_indices(x)
        return ret
    elif "-" in txt:
        start, end = txt.split("-")
        return list(range(int(start), int(end) + 1))
    else:
        return [int(txt)]


def hydra_script_runner(
    script_level: int,
    setting_name: str,
    overrides: List[str],
    ground_truth_yaml_directory: Path,
    generated_yaml_directory: Path,
    dummy: bool,
    main_fn: Callable,
    script_name: str,
    exclude_attributes: List[str] | None = None,
):
    """
    This function runs a hydra test by invoking a main function `main_fn`
    which is in either of the provided scripts of the project. It also
    has two settings:

    1. dummy = True:
        In this scenario, only the hydra resolver is called and the resulting
        yaml is compared against the reference ground truth yaml which is already
        stored in the reference file. This does not invoke any logic and is
        fast to test.
    2. dummy = False:
        In this scenario, the configuration is fed into the main_fn and also
        the logs will be printed out to mlflow.

    Args:

        script_level:
            an indicator of how hard a test is. This is used for skipping.
            If the SCRIPT_LEVEL environment variable that is specifies does not
            contain that script level, then it would be skipped.

        setting_name:
            This is a name given to a particular setting of the configuration and
            is used for logging purposes.
            For example,
            ```
            python scripts/train.py dataset=mnist +experiment=train_diffusion_greyscale
            ```
            might get a name: `train_mnist_diffusion`

        overrides:
            The sequence of hydra overrides that the script goes through. For example,
            ```
            python scripts/train.py dataset=mnist +experiment=train_diffusion_greyscale
            ```
            has the following sequence of overrides:
            ```
            [dataset=mnist, +experiment=train_diffusion_greyscale]
            ```

        ground_truth_yaml_directory:
            This is a directory that contains different yaml files (with the naming convention
            setting_name.yaml) which store the yamls that should be parsed.

        generated_yaml_directory:
            This is a directory where the parsed yaml should be dumped into. The diff between
            the parsed yaml `setting_name.yaml` in this directory and the same in the
            ground_truth_yaml_directory are of interest when a test fails.

        dummy:
            Whether or not to actually run the `main_fn` on the parsed yaml.

        main_fn:
            A function that contains the logic of the script; these functions are main() functions
            in the scripts/ directory.

        script_name:
            The name associated with the script. For example, script/train.py would have the name
            "train".

        exclude_attributes:
            These are the attributes to be excluded while comparing the configurations.
            Things like timestamps, and other attributes that are not deterministic should be
            excluded.

    """
    env_var = os.getenv("SCRIPT_LEVEL", default="ALL")
    level_indices = []
    all_selected = False
    if env_var == "ALL":
        all_selected = True
    else:
        level_indices = parse_test_indices(env_var)

    # if enabled, then the tests where the logic takes place will be run
    enable_mlflow_logging = os.getenv("ENABLE_MLFLOW_LOGGING", default="False") == "True"

    with initialize(version_base=None, config_path="../../conf"):
        dotenv.load_dotenv(override=True)  # required for configuration variables in the environment
        cfg = compose(config_name=script_name, overrides=overrides, return_hydra_config=True)
        HydraConfig().cfg = cfg
        OmegaConf.resolve(cfg)
        if all_selected or script_level in level_indices:
            if dummy:
                cfg_filtered = tools.filter_cfg(
                    cfg, exclude_attributes=(exclude_attributes or []) + ["hydra"]
                )
                yml_text = OmegaConf.to_yaml(cfg_filtered)
                with open(generated_yaml_directory / f"{setting_name}.yaml", "w") as f:
                    f.write(yml_text)
                # read from the ground truth file and store in a variable
                yml_ground_truth = (
                    ground_truth_yaml_directory / f"{setting_name}.yaml"
                ).read_text()
                assert (
                    yml_text == yml_ground_truth
                ), f"YAMLs do not match! Check {ground_truth_yaml_directory / f'{setting_name}.yaml'} and {generated_yaml_directory / f'{setting_name}.yaml'}"

            else:
                if enable_mlflow_logging:
                    main_fn(cfg)
                else:
                    pytest.skip(
                        "Skipping mlflow logging test, set ENABLE_MLFLOW_LOGGING to True to run"
                    )
        else:
            pytest.skip(
                f"Skipping level {script_level} test, add {script_level} to SCRIPT_LEVEL to run"
            )
