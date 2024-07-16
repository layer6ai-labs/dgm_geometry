"""
This script automatically downloads all the resources and checkpoints that are required
for reproducing the results. 

We also set up the appropriate dotenv environment variables to a default value for the 
user that they can modify if they wish to later.
"""

import os
import shutil
from pathlib import Path

import gdown
import dotenv

if os.environ.get("IS_TESTING", False):
    from scripts import tools
else:
    import tools


def main(download_files: bool = True):
    """
    This function downloads all the resources and checkpoints required for the project.
    If download_files is set to True, then it will prompt the user every time for downloading the files.
    Otherwise, it will assume that the files are already downloaded and will set the environment variables
    and check if the files exist.
    """
    dotenv_file = dotenv.find_dotenv()
    # if the dotenv file does not exist, create it
    if dotenv_file == "":
        # make an empty dotenv file
        dotenv_file = os.path.join(os.getcwd(), ".env")
        with open(dotenv_file, "w") as f:
            f.write("")

    # create a copy of dotenv_file
    mex = 0
    while os.path.exists(f".env_backup{mex}"):
        mex += 1
    shutil.copy(dotenv_file, f".env_backup{mex}")

    dotenv.load_dotenv(dotenv_file, override=True)

    os.makedirs("outputs/downloads/dgm-geometry", exist_ok=True)

    # get the current absolute path
    project_path = Path(os.getcwd()).absolute()

    # (1) download the checkpoints for generative models and set the dotenv variables
    if download_files:
        resp = input(
            "Do you wish to download the checkpoint files? (y/n)\nYou can press 'n' to exit ... "
        )
        if resp.lower() == "y":
            gdown.download_folder(
                "https://drive.google.com/drive/u/1/folders/1n8fZj2wajUk5By9ZuWnK6gRNLXQPC7l6",
                output="outputs/downloads/dgm-geometry/",
            )

    # (1.1) download diffusion checkpoints, both with MLPs and with ConvNet UNets
    print("Setting environment variables for diffusion checkpoints ...")
    for checkpoint_key, checkpoint_path in [
        ("DIFFUSION_CIFAR10_CHECKPOINT", os.path.join("cifar10", "epoch=456-step=160864.ckpt")),
        ("DIFFUSION_SVHN_CHECKPOINT", os.path.join("svhn", "epoch=610-step=350103.ckpt")),
        ("DIFFUSION_FMNIST_CHECKPOINT", os.path.join("fmnist", "epoch=540-step=253729.ckpt")),
        ("DIFFUSION_MNIST_CHECKPOINT", os.path.join("mnist", "epoch=453-step=212926.ckpt")),
        (
            "DIFFUSION_CIFAR10_MLP_CHECKPOINT",
            os.path.join("cifar10-mlp", "epoch=1192-step=466463.ckpt"),
        ),
        ("DIFFUSION_SVHN_MLP_CHECKPOINT", os.path.join("svhn-mlp", "epoch=1045-step=509450.ckpt")),
        ("DIFFUSION_MNIST_MLP_CHECKPOINT", os.path.join("mnist-mlp", "epoch=780-step=366289.ckpt")),
        (
            "DIFFUSION_FMNIST_MLP_CHECKPOINT",
            os.path.join("fmnist-mlp", "epoch=995-step=467124.ckpt"),
        ),
    ]:
        os.environ[checkpoint_key] = os.path.join(
            project_path,
            "outputs",
            "downloads",
            "dgm-geometry",
            "checkpoints",
            "diffusions",
            checkpoint_path,
        )

        # check if the os.environ[checkpoint_key] file exists
        if not os.path.exists(os.environ[checkpoint_key]):
            raise FileNotFoundError(
                f"Could not find the checkpoint file at {os.environ[checkpoint_key]}"
            )

        dotenv.set_key(dotenv_file, checkpoint_key, os.environ[checkpoint_key])

        # parse the checkpoint path and find a epoch=?:
        epoch = int(checkpoint_path.split("epoch=")[1].split("-")[0]) + 2
        os.environ[checkpoint_key + "_N_EPOCH"] = str(epoch)
        dotenv.set_key(
            dotenv_file, checkpoint_key + "_N_EPOCH", os.environ[checkpoint_key + "_N_EPOCH"]
        )
    print("[x] checked diffusion checkpoints")

    # (1.1) download the flow checkpoints
    print("Setting environment variables for diffusion checkpoints ...")
    for checkpoint_key, checkpoint_path in [
        ("FLOW_MNIST_CHECKPOINT", os.path.join("mnist", "epoch=196-step=92393.ckpt")),
        ("FLOW_FMNIST_CHECKPOINT", os.path.join("fmnist", "epoch=216-step=101773.ckpt")),
        ("FLOW_CIFAR10_CHECKPOINT", os.path.join("cifar10", "epoch=201-step=78982.ckpt")),
        ("FLOW_SVHN_CHECKPOINT", os.path.join("svhn", "epoch=191-step=110016.ckpt")),
    ]:
        os.environ[checkpoint_key] = os.path.join(
            project_path,
            "outputs",
            "downloads",
            "dgm-geometry",
            "checkpoints",
            "flows",
            checkpoint_path,
        )

        # check if the os.environ[checkpoint_key] file exists
        if not os.path.exists(os.environ[checkpoint_key]):
            raise FileNotFoundError(
                f"Could not find the checkpoint file at {os.environ[checkpoint_key]}"
            )

        dotenv.set_key(dotenv_file, checkpoint_key, os.environ[checkpoint_key])

        # parse the checkpoint path and find a epoch=?:
        epoch = int(checkpoint_path.split("epoch=")[1].split("-")[0]) + 2
        os.environ[checkpoint_key + "_N_EPOCH"] = str(epoch)
        dotenv.set_key(
            dotenv_file, checkpoint_key + "_N_EPOCH", os.environ[checkpoint_key + "_N_EPOCH"]
        )
    print("[x] checked flow checkpoints!")

    # (2) download the OOD report files and set the appropriate environment variables
    if download_files:
        resp = input("Do you wish to download the OOD files? (y/n)\nYou can press 'n' to exit ... ")
        if resp.lower() == "y":
            gdown.download_folder(
                "https://drive.google.com/drive/u/1/folders/1xIPuBzsh495Y9WnviQwaVpZZ_oJdVMJ5",
                output="outputs/downloads/dgm-geometry/",
            )

    for env_key in [
        "OOD_LIKELIHOOD_PARADOX_FLOW_MNIST",
        "OOD_LIKELIHOOD_PARADOX_FLOW_FMNIST",
        "OOD_LIKELIHOOD_PARADOX_DIFFUSION_FMNIST",
        "OOD_LIKELIHOOD_PARADOX_DIFFUSION_MNIST",
        "OOD_LID_CURVE_FLOW_FMNIST",
        "OOD_LID_CURVE_FLOW_MNIST",
        "OOD_LID_CURVE_DIFFUSION_FMNIST",
        "OOD_LID_CURVE_DIFFUSION_MNIST",
        "OOD_LID_LIKELIHOOD_FLOW_FMNIST_1",
        "OOD_LID_LIKELIHOOD_FLOW_MNIST_1",
        "OOD_LID_LIKELIHOOD_DIFFUSION_FMNIST_1",
        "OOD_LID_LIKELIHOOD_DIFFUSION_MNIST_1",
        "OOD_LID_LIKELIHOOD_FLOW_FMNIST_2",
        "OOD_LID_LIKELIHOOD_FLOW_MNIST_2",
    ]:
        os.environ[env_key] = os.path.join(
            project_path, "outputs", "downloads", "dgm-geometry", "ood_reports", env_key.lower()
        )
        if not os.path.exists(os.environ[env_key]):
            raise FileNotFoundError(f"Could not find the report file at {os.environ[env_key]}")
        dotenv.set_key(dotenv_file, env_key, os.environ[env_key])
    print("[x] checked OOD detection checkpoints!")


DISCLAIMER = """
[Disclaimer]
\tThis script will download all the resources and checkpoints required for the project, 
\tand will replace your existing .env file with the new one. This is only recommended 
\tto run once to set up the project.

\tNote that this script will also backup your existing .env file to .env_backup{mex} 
\twhere mex is the number of backups already present in case you need to revert back.
"""

if __name__ == "__main__":
    # setting up root to be the root of the project
    tools.setup_root()
    print(DISCLAIMER)
    resp = input("Do you wish to continue? (y/n)\nYou can press 'n' to exit ... ")
    setup = resp.lower()[0] == "y"
    if not setup:
        print("Exiting ...")
        exit(0)
    print("Continuing ...")

    if os.environ.get("IS_TESTING", False):
        main(download_files=False)
    else:
        main()
