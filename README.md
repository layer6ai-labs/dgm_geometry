# DGM Geometry

Here, we study how the geometry of deep generative models (DGMs) can inform our understanding of phenomena like OOD detection. In tandem and as a supplement to these topics, we also study algorithms for local intrinsic dimension (LID) estimation of datapoints.

## Installation

We use a conda environment for this project. To create the environment, run:
```bash
conda env create -f env.yaml
# this will create an environment named dgm_geometry
conda activate dgm_geometry
```

To download all the checkpoints, resources, and setup appropriate environment variables, run the following:
```bash
python scripts/download_resources.py
```
You may choose to skip this stage if you want to train your own models, but it is recommended as some of the notebooks.

## Training a Deep Generative Model

Most of the capabilities in the codebase involve using the training script, we use [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) for training and [lightning callbacks](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html) for monitoring the behaviour and properties of the manifold induced by the generative model. Even when no training is involved, we use the training script but load checkpoints and set the epoch count to zero.

Training involves running `scripts/train.py` alongside a dataset and an experiment configuraton. To get started, you can run the following examples for training flows or diffusions on image datasets:

```bash
# to train a greyscale diffusion, run the following! You can for example replace the dataset argument with mnist or fmnist
python scripts/train.py dataset=<grayscale-data> +experiment=train_diffusion_greyscale
# to train an RGB diffusion, run the following! You can for example replace the dataset argument with cifar10
python scripts/train.py dataset=<rgb-data> +experiment=train_diffusion_rgb
# to train a greyscale flow, run the following! You can for example replace the dataset argument with mnist or fmnist
python scripts/train.py dataset=<grayscale-data> +experiment=train_flow_greyscale
# to train an RGB flow, run the following! You can for example replace the dataset argument with cifar10
python scripts/train.py dataset=<rgb-data> +experiment=train_flow_rgb
```
For example:
```bash
python scripts/train.py dataset=mnist +experiment=train_diffusion_greyscale
```

### Tracking

We use [mlflow](https://mlflow.org/) for tracking and logging; all the artifactors will be available in the `outputs` subdirectory. To set up mlflow, run the following:

```bash
cd outputs
mlflow ui
```
When ran, you can click on the provided link to view all the experiments. The logs are typically stored in the artifacts directory.

### Test Runs

Use the following script to see the configuration that the script ends up running:
```bash
python scripts/train.py <training-options> --help --resolve
```
To perform a test run in development mode, you can run the following:
```bash
python scripts/train.py <training-options> dev_run=true train.trainer.callbacks=null train.trainer.fast_dev_run=True
```
For example:
```bash
python scripts/train.py dataset=cifar10 +experiment=train_diffusion_rgb --help --resolve # show configurations
python scripts/train.py dataset=cifar10 +experiment=train_diffusion_rgb dev_run=true train.trainer.callbacks=null train.trainer.fast_dev_run=True # run without trainig logic
```

## Maintaining

### Simple Tests
Please sort imports, format the code, and run the tests before pushing any changes:
```bash
isort data lid models tests
black -l 100 .
pytest tests
```
### Hydra Tests
In addition, plase ensure that the hydra scripts are also working as expected. By default, the pytests command will only check the barebone YAML files for the Hydra scripts. Before merging major PRs, please run the following command which serves as an integration test. It will take some time but it will ensure that all the scripts are backwards compatible:
```bash
ENABLE_MLFLOW_LOGGING=True pytest tests/hydra
```
If, for example, the tests fail on specific settings, you can test them individually by setting the `SCRIPT_LEVEL` variable. Example include:
```bash
SCRIPT_LEVEL=ALL ENABLE_MLFLOW_LOGGING=True pytest tests/hydra/<hydra-test-file> # to run all the scripts
SCRIPT_LEVEL=0 ENABLE_MLFLOW_LOGGING=True pytest tests/hydra/<hydra-test-file> # to run a specific script
SCRIPT_LEVEL=0,2 ENABLE_MLFLOW_LOGGING=True pytest tests/hydra/<hydra-test-file> # to run multiple scripts
SCRIPT_LEVEL=0-2 ENABLE_MLFLOW_LOGGING=True pytest tests/hydra/<hydra-test-file> # to run a range of scripts
SCRIPT_LEVEL=0,2,3-5,7-10 ENABLE_MLFLOW_LOGGING=True pytest tests/hydra/<hydra-test-file> # to run multiple ranges of scripts
```
This mechanism is incorporated to allow for more granular control over the tests. As an example, when you encounter errors, pytest will show you an error on setting `[True,setting{idx}]`, and in turn, you can run the script with `SCRIPT_LEVEL=idx` to debug the error associated with that script. For a full list of all the scripts that are being used for testing, please look at the corresponding script's test directory under `tests/hydra`.

### Additional Notes on Hydra Tests

Note that even without setting this variables, the barebone resolved configurations are compared to ground truth configurations stored in `tests/resources/hydra_config`. If you want to add a new configuration, please add it to the `tests/resources/hydra_config` directory and then run the tests to ensure that the configurations are correct. When test configurations are being compared, the current version of the resolved configuration can be found under `outputs/hydra_config`, you may want to compare your configurations in that directory with the ones in `tests/resources/hydra_config` to ensure that the configurations are correct. This will be automatically done when you run the tests. In addition, you can also qualitatively monitor the runs by openning up your mlflow server and looking at the new runs in `hydra_config`. These runs are tagged using the current data and time and the setting (look for the tags `setting` and `timestamp` for these runs).


## Maintaining the Website

This repository also hosts the content for the website related to these projects. The website consists of some `html` files and runnable `jupyter` notebooks.
All of our notebooks are maintained in our website which works with a Quarto plugin. 
You may update the content of the website by changing the notebooks and `.qmd` files `docs/` directory. 
To see the updates in real-time, run the following command that will start a local server:
```bash
# download and install Quarto from https://quarto.org/docs/get-started/
cd docs
quarto preview # opens up a local server on port 4200
```
