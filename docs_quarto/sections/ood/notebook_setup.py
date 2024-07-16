"""
Script meant to be imported at the beginning of experimental notebooks.
Sets random seeds and moves the notebook's working directory to the project root.
"""

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Configure file system
project_root = Path(__file__).parent
while True:
    if (
        (project_root / "lid").exists()
        and (project_root / "data").exists()
        and (project_root / "scripts").exists()
    ):
        break
    project_root = project_root.parent
data_path = project_root / "data"
os.chdir(project_root)

# Configure PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(0)

# Set matplotlib colour theme
plt.style.use("seaborn-v0_8-pastel")
