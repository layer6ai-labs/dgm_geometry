# testing the interface of model_based LID


import numpy as np
import pytest
import torch

from lid.diffusions import CFDM_LID

torch.manual_seed(0)


# This is a very strong test but also takes some time to run!
@pytest.mark.parametrize(
    "setting",
    [
        (42, [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1, 1], 0.1, 1e-3, 2),
        (100, [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1, 1, 1], 0.2, 1e-3, 3),
        (111, [1e-6, 1e-6, 1e-6, 1, 1, 1], 0.2, 1e-3, 3),
    ],
)
def test_fokker_planck(setting):
    seed, cov_eigs, tolerance, eps, true_lid = setting
    torch.manual_seed(seed)
    device = torch.device("cpu")
    n_samples = 10000
    d = len(cov_eigs)
    mean = d * torch.randn(d).to(device)

    # create a covariance matrix that has an almost zero eigenvalue, 3 larger eigenvalues and 6 smaller eigenvalues
    eigvals = torch.tensor(cov_eigs).to(device)
    # create a random orthogonal matrix
    orthogonal = torch.randn(d, d).to(device)
    q, _ = torch.linalg.qr(orthogonal)
    cov = q @ torch.diag(eigvals) @ q.T
    # take the eigenvalues of cov
    eigvals = torch.linalg.eigvalsh(cov)

    # take samples from isotropic Gaussian of dimension d
    isotropic_gaussian = torch.randn(n_samples, d).to(device)
    # multiply each element of isotropic_gaussian by the square root of the eigenvalues of cov
    gaussian_scaled = isotropic_gaussian * torch.sqrt(eigvals)
    # transform them to have covariance cov
    transformed = gaussian_scaled @ q.T
    # transfoem them to have mean mean
    data = transformed + mean

    lid_estimator = CFDM_LID(
        data=data,
        ambient_dim=d,
        device=device,
        beta_min=0.1,
        beta_max=20,
        t_max=1.0,
    )
    artifact = lid_estimator.preprocess(data)
    # find bulge
    all_lid = []
    for t in np.linspace(eps, 1, 10):
        lid = lid_estimator.compute_lid_from_artifact(artifact, t=t).mean()
        all_lid.append(lid)
    mx_lid = max(all_lid)
    diff = abs(true_lid - mx_lid)
    assert (
        diff < tolerance
    ), f"The LID should be close to the true LID but got {mx_lid} and {true_lid}"
