"""
This code defines a decorator that sets up mlflow for an experiment.

For example,

@tools.MlflowDecorator(

)
def main(cfg):
    ...

"""

import sys
import contextlib
from typing import Any, Callable, List, Dict
import functools
from pathlib import Path
import dotenv

import mlflow
from omegaconf import DictConfig, OmegaConf


class Tee:
    """This class allows for redirecting of stdout and stderr"""

    def __init__(self, primary_file, secondary_file):
        self.primary_file = primary_file
        self.secondary_file = secondary_file

        self.encoding = self.primary_file.encoding

    # TODO: Should redirect all attrs to primary_file if not found here.
    def isatty(self):
        return self.primary_file.isatty()

    def fileno(self):
        return self.primary_file.fileno()

    def write(self, data):
        # We get problems with ipdb if we don't do this:
        if isinstance(data, bytes):
            data = data.decode()

        self.primary_file.write(data)
        self.secondary_file.write(data)

    def flush(self):
        self.primary_file.flush()
        self.secondary_file.flush()


@contextlib.contextmanager
def link_output_streams(artifact_dir: Path):
    out_file = open(artifact_dir / "stdout.txt", "a")
    err_file = open(artifact_dir / "stderr.txt", "a")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = Tee(primary_file=sys.stdout, secondary_file=out_file)
    sys.stderr = Tee(primary_file=sys.stderr, secondary_file=err_file)

    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        out_file.close()
        err_file.close()


def filter_cfg(
    cfg,
    include_attributes: List[str] | str | None = None,
    exclude_attributes: List[str] | str | None = None,
):

    if include_attributes is not None:
        if not isinstance(include_attributes, list):
            include_attributes = [include_attributes]
        # Create a new empty configuration to hold the included keys
        cfg_ret = OmegaConf.create()

        for attribute_path in include_attributes:
            attr_list = attribute_path.split(".")
            cfg_cur = cfg_ret
            cfg_cur_ref = cfg
            for attr in attr_list[:-1]:
                if attr not in cfg_cur:
                    cfg_cur = {}
                cfg_cur = cfg_cur[attr]
                cfg_cur_ref = cfg_cur_ref[attr]

            cfg_cur[attr_list[-1]] = cfg_cur_ref[attr_list[-1]]

    elif exclude_attributes is not None:
        if isinstance(cfg, dict):
            cfg_dict = cfg
        else:
            cfg_dict = OmegaConf.to_container(cfg)
        cfg_ret = OmegaConf.create(cfg_dict)
        if not isinstance(exclude_attributes, list):
            exclude_attributes = [exclude_attributes]

        for attribute_path in exclude_attributes:
            attr_list = attribute_path.split(".")
            path_successful = True
            cfg_cur: DictConfig = cfg_ret
            for attr in attr_list[:-1]:
                if attr not in cfg_cur:
                    path_successful = False
                    break
                cfg_cur = cfg_cur[attr]
            if path_successful and attr_list[-1] in cfg_cur:
                del cfg_cur[attr_list[-1]]

    else:
        if isinstance(cfg, dict):
            cfg_dict = cfg
        else:
            cfg_dict = OmegaConf.to_container(cfg)
        cfg_ret = OmegaConf.create(cfg_dict)

    return cfg_ret


class MlflowDecorator:
    """
    This is a decorator that handles setting up scripts for the project.
    At a high-level, this script sets up mlflow and passes in an artifact_directory
    where the main script can log into.

    We decorate our main function using a combination of this decorator
    and the hydra decorator:

    @hydra.main(...)
    @MlflowDecorator(...)
    main(cfg, artifact_dir):
        ...

    Note that the artifact_dir here would be the directory in which all the data will be logged.

    This decorator is designed to be coupled with the hydra configuration and will also log
    the hydra configuration (cfg) into mlflow for better reproducibility. The file will be called
    `config.yaml` and will contain all the configurations after the hydra resolution.

    If there are keys or private information being added to the hydra configuration file that you
    do not want to log, you can choose to remove them using the `exclude_attributes` that is explained
    below. You may also choose to only include a subset of attributes using `include_attributes`.

    Moreover, this decorator also links stderr and stdout to stderr.txt and stdout.txt on mlflow
    so that you can check out all the output stream of the run.

    For a better understanding, we encourage you to check out one of our scripts that uses this
    decorator in the scripts/.. directory.
    """

    def __init__(
        self,
        out_dir: str = "./outputs",
        experiment_name: str = "Default",
        pytorch_autolog: Dict | None = None,
        include_attributes: List[str] | None = None,
        exclude_attributes: List[str] | None = None,
        tags: Dict[str, Any] | None = None,
    ):
        """
        Args:
            out_dir:
                The directory under which the mlflow logs will be created. This
                refers to `mlruns` and means that we would have `out_dir/mlruns`.
            experiment_name:
                The experiment name or group where all the logs will be stored at.
                All experiments with the same name will be categorized into the same
                tab in the mlflow ui. Note that cfg.mlflow.experiment_name will override
                this argument if specified in the configuration.
            pytorch_autolog:
                The kwargs to pass to mlflow.pytorch.autolog(...) if there are any
                specific model logging that needs to be done. Note that cfg.mlfllow.pytorch_autolog
                will override this argument if specified in the configuration.
            tags:
                Some preset tags. Note that cfg.mlflow.tags will override this argument
                if specified in the configuration.
            include_attributes:
                This refers to the sole attributes to keep when logging the Hydra cfg into
                `config.yaml`. Note that if you for example want to include all of attribute
                'a' and only the sub-attribute 'c' of attribute 'b' you can do the following:
                include_attributes = ['a', 'b.c']
            exclude_attributes:
                This refers to the attrbutes to exclude from cfg when logging to config.yaml.
                This can include private information, or attributes that hydra uses internally
                to resolve the configuration. Note that if you want to exclude all of attribute
                'a' and the sub-attribute 'c' of attribute 'b' you can do the following:
                exclude_attributes = ['a', 'b.c']
        """
        self.out_dir = out_dir
        self.experiment_name = experiment_name
        self.pytorch_autolog = pytorch_autolog or {}
        self.tags = tags or {}

        self.include_attributes = include_attributes
        self.exclude_attributes = exclude_attributes
        assert (
            include_attributes is None or exclude_attributes is None
        ), "Attributes clash, only specify one of include_attributes or exclude_attributes."

    def __call__(self, main_func: Callable) -> Callable:

        def wrapper(cfg: DictConfig) -> Any:
            # load the environment variables from your local '.env'
            # Required for config variables that need to be resolved using the environment
            dotenv.load_dotenv(override=True)

            # set out_dir according to cfg, and if not availble, set it according
            # to the attributes of the decorator
            out_dir = Path(cfg.get("out_dir", self.out_dir))
            # set the mlflow attributes according to cfg, and if not availble, set it according
            # to the attributes of the decorator
            mlflow_attributes = cfg.get("mlflow", {})
            experiment_name = mlflow_attributes.get("experiment_name", self.experiment_name)
            pytorch_autolog_kwargs = mlflow_attributes.get("pytorch_autolog", self.pytorch_autolog)
            tags = mlflow_attributes.get("tags", self.tags)
            # filter the cfg attributes for logging into mlflow
            cfg_filtered = filter_cfg(
                OmegaConf.to_container(cfg, resolve=True),
                include_attributes=self.include_attributes,
                exclude_attributes=self.exclude_attributes,
            )
            cfg_filtered = filter_cfg(
                cfg_filtered,
                exclude_attributes=["hydra", "Defaults"],
            )
            # Set up MLflow logging only if dev_run is set to False
            if not cfg.dev_run:
                mlflow.set_tracking_uri(out_dir / "mlruns")
                mlflow.set_experiment(experiment_name)
                mlflow.start_run()
                mlflow.pytorch.autolog(**pytorch_autolog_kwargs)
                artifact_dir = out_dir / mlflow.get_artifact_uri().split(":")[1]
                mlflow.log_text(OmegaConf.to_yaml(cfg_filtered), "config.yaml")
            else:
                artifact_dir = None

            try:
                # link stderr and stdout to the artifact directory of mlflow
                # so that we can monitor the outputs from mlflow
                with link_output_streams(artifact_dir=artifact_dir):
                    ret = main_func(cfg, artifact_dir)
                    # set the tags once the run is complete to consider for resolutions
                    mlflow.set_tags(tags)
            finally:
                if not cfg.dev_run:
                    mlflow.end_run()

            return ret

        return wrapper
