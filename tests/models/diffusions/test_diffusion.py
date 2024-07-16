import lightning as L
import pytest
import torch

from data.transforms.unpack import UnpackBatch
from models.diffusions.networks import SimpleDiffusionMLP
from models.diffusions.sdes import VpSde
from models.diffusions.training import LightningDiffusion


@pytest.fixture
def lit_diffusion():
    network = SimpleDiffusionMLP(pytest.ambient_dim)
    sde = VpSde(network)
    unpack = UnpackBatch(access_tokens=[0])
    return LightningDiffusion(sde, unpack_batch=unpack)


def test_sample(lit_diffusion):
    sample = lit_diffusion.sample(6, sample_shape=(6,), batch_size=4)
    assert sample.shape == ((6, 6))
    assert not sample.isnan().any()


@pytest.mark.filterwarnings("ignore:GPU available but not used.")
@pytest.mark.filterwarnings(
    "ignore:You defined a `validation_step` but have no `val_dataloader`. Skipping val loop."
)
@pytest.mark.filterwarnings(
    "ignore:The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=127` in the `DataLoader` to improve performance"
)
def test_training(lit_diffusion, gaussian_gt):
    mu, cov = gaussian_gt
    data = torch.randn(32, pytest.ambient_dim) * torch.sqrt(cov) + mu
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset)

    trainer = L.Trainer(
        devices=1,
        accelerator="cpu",
        fast_dev_run=True,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model=lit_diffusion, train_dataloaders=dataloader)
