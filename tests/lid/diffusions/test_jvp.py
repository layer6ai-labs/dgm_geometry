import pytest
import torch
import torch.utils
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset

from data.transforms.unpack import UnpackTabular
from models.diffusions.networks import AttnScoreNetwork, MLPUnet
from models.diffusions.sdes import VpSde
from models.diffusions.training import LightningDiffusion
from models.training import LightweightTrainer

AMBIENT_DIM = 10


@pytest.mark.parametrize(
    "device",
    ["cpu", "cuda"],
)
@pytest.mark.parametrize(
    "score_net",
    [
        AttnScoreNetwork(d=AMBIENT_DIM, k=8, L=1, num_heads=2, dim_feedforward=20),
        MLPUnet(data_dim=AMBIENT_DIM, time_embedding_dim=32, hidden_sizes=[32, 32]),
    ],
)
def test_jvp(device, score_net):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # if the score network is an instance of AttnScoreNetwork, skip the test
    # since it is not implemented for this network
    if isinstance(score_net, AttnScoreNetwork):
        pytest.skip(
            f"AttnScoreNetwork does not have a forward AD in the current version of pytorch: {torch.__version__}"
        )
    device = torch.device(device)
    data = torch.randn(8, AMBIENT_DIM)
    data = data.to(device)
    score_net = score_net.to(device)
    vpsde = VpSde(score_net=score_net)
    model = LightningDiffusion(
        sde=vpsde,
        optim_partial=lambda p: torch.optim.Adam(p, lr=1e-3),
        unpack_batch=UnpackTabular(),
    )
    trainer = LightweightTrainer(
        max_epochs=10,
        device=device,
    )
    trainer.fit(
        model,
        TorchDataLoader(TensorDataset(data), batch_size=2, shuffle=True),
    )
    vpsde.eval()
    score_net.eval()
