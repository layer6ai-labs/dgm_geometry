import copy
import functools
import math
from typing import Callable, List, Literal, Tuple

import torch
from tqdm import tqdm

from models.flows.diffeomorphisms import Diffeomorphism, Sinusoidal

from .lid_base import LIDDistribution


class ManifoldMixture(LIDDistribution):
    """
    A distribution over a mixture of manifolds of different dimensions.

    The distribution is defined as follows:
    1. A set of modes are defined in the ambient space and each mode is associated with a manifold.
    2. For each mode, there is a probability of selecting that mode.
    3. The distribution on each manifold is defined via chaining three transformations:
        a. A diffeomorphism that projects a simple distribution in the latent space to a more complex one.
        b. A affine transformation that projects the distribution to a higher dimensional space.
        c. A diffeomorphism that projects the distribution in the ambient space to a more complex one.
    4. The distribution is normalized to ensure that the marginals are normalized around the modes.

    **Generic diffeomorphisms for various manifold modelling**: You can pass in a list of diffeomorphism partials that construct
    diffeomorphisms in latent and ambient space.
    [
        (diffemorphism_partial_latent_1, diffeomorphism_partial_ambient_1),
        (diffemorphism_partial_latent_2, diffeomorphism_partial_ambient_2),
        ...
    ]
    Each partial takes in the number of dimensions and returns a diffeomorphism inheriting from the Diffeomorphism class.
    If the partial is None, then the identity map is applied in that space instead.
    These diffeomorphisms are generic and can be defined using any normalizing flow transform.
    In fact, if the diffeomorphisms are torch modules, then they are trained to be stable by adjusting their condition number.

    **Condition number control**: The distribution also controls for ill-conditioned transformations by training
    the diffeomorphisms to be stable around the generated samples. For a set of samples 'z', a diffeomorphism 'f' is
    trained to minimize the following loss:

    loss = sum_ij (kappa_ij + 1 / kappa_ij - 2) * mask_ij * penalty / sum_ij(mask_ij)
    where 'kappa_ij' = ||f(z_i) - f(z_j))||_2 / ||z_i - z_j||_2 + eps
    and 'mask_ij' is a binary mask that is 1 if the distance between 'z_i' and 'z_j' AND 'f(z_i)' and 'f(z_j)'
    is less than a threshold. In the code, the threshold is controlled by 'condition_number_neighbourhood_threshold'.
    At optimality, the function becomes conformal for the neighbourhoods defined using this threshold because kappa_ij=1.

    Note that all of this happens if 'adjust_condition_number' is set to True, if not, then the diffeomorphisms are trained
    otherwise they are left untrained.

    **Affine projection**: The distribution also supports different types of affine projections:
    1. Random: A random linear transformation is applied to the distribution, the linear transform has elements drawn from a normal distribution.
    2. Repeat: The columns of the distribution are repeated to make it higher dimensional.
    3. Zero-pad: The columns of the distribution are zero-padded to make it higher dimensional.
    4. Random-rotation: A random orthogonal matrix is applied to the distribution.
    No condition number adjustment is done for the affine projection, thus, make sure that the affine projection
    does conform to the condition number constraints.

    **Suggestions**: Generally, avoid using diffeomorphisms that are too complex, as they can lead to numerical instability, slow sampling,
    and also poor condition number. Even the condition number adjustment can only do so much, so it is better to keep the diffeomorphisms simple.
    One good diffeomorphism to use is the additive coupling transform, which is simple and stable and has a controlled condition number.

    """

    TRAINING_PATIENCE = 10
    LOSS_AVG_WINDOW_SIZE = 5
    TOLERANCE = 0.1
    BIN_SEARCH_LR_N_ITER = 10

    DiffeomorphicConstructor = Callable[
        [int], Diffeomorphism
    ]  # Type hint for the diffeomorphism constructor

    def _init_mixture_probs(self, mixture_probs: List[float] | None = None):
        """
        Turn a list of mixture probabilities into a tensor and normalize it, and then compute the CDF for future indexing and sampling.
        """
        # Handle mixture_probs
        mixture_probs = mixture_probs or [1.0] * self.n_mixtures
        assert len(mixture_probs) == len(
            self.manifold_dims
        ), f"The length of mixture_probs and manifold_dims should match! ({len(mixture_probs)} != {len(self.manifold_dims)})"
        self.mixture_probs = torch.tensor(mixture_probs)
        self.mixture_probs /= torch.sum(self.mixture_probs)
        self.mixture_probs_cdf = torch.cumsum(self.mixture_probs, dim=0).to(self.device)

    def _init_modes(self, ambient_dim: int, distance_between_modes: float):
        """
        If the number of mixtures are less than the ambient dimension + 1, then a simplex is created in the ambient space.
        And the modes are taken from the vertices of the simplex.

        Otherwise, equidistance points on a circle are generated and then projected to the ambient space.

        Note that distance_between_modes is the distance between the modes in the ambient space.
        """
        if self.n_mixtures <= ambient_dim:
            # create a simplex in the ambient space
            simplex = (torch.eye(ambient_dim) - 0.5) / torch.sqrt(torch.tensor(2.0))
            # now find the projection of the simplex in the ambient space

            self.modes = simplex[: self.n_mixtures] * distance_between_modes
            self.modes = self.modes.to(self.device)
            return
        thetas = torch.linspace(0, 2 * torch.pi, self.n_mixtures + 1)[:-1]
        L_radius = distance_between_modes / 2
        R_radius = self.n_mixtures * distance_between_modes
        for _ in range(100):
            radius = (L_radius + R_radius) / 2
            x = math.cos(thetas[1]) * radius
            y = math.sin(thetas[1]) * radius
            if ((x - 1) ** 2 + y**2) ** 0.5 > distance_between_modes:
                R_radius = radius
            else:
                L_radius = radius
        radius = R_radius
        self.modes = torch.stack([radius * torch.cos(thetas), radius * torch.sin(thetas)], dim=1)
        # zero-pad to the ambient dimension
        if ambient_dim > 2:
            self.modes = torch.cat(
                [
                    self.modes,
                    torch.zeros(self.n_mixtures, ambient_dim - 2, dtype=self.modes.dtype),
                ],
                dim=1,
            )

        self.modes = self.modes.to(self.device)

    def _sample_latent(
        self,
        n_samples,
        dim,
        noise_type: Literal["normal", "uniform", "laplace"] | None = None,
        seed: int | None = None,
    ):
        """
        Sample in the latent space, which is later used to project onto the ambient space.
        These samples are drawn from a distribution specified by `sample_distr` in the constructor.

        Args:
            n_samples (int): Number of samples to draw
            dim (int): Dimensionality of the latent space
            seed (int | None): Seed for reproducibility
        Returns:
            torch.Tensor: Samples of shape (n_samples, dim)
        """
        seed = seed or torch.randint(0, 100000, (1,)).item()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            noise_type = noise_type or self.sample_distr
            if noise_type.startswith("normal"):
                ret = torch.randn(n_samples, 1 if noise_type.endswith("repeated") else dim).to(
                    self.device
                )
            elif noise_type.startswith("uniform"):
                ret = (
                    torch.rand(n_samples, 1 if noise_type.endswith("repeated") else dim).to(
                        self.device
                    )
                    * 2
                    - 1
                )
            elif noise_type.startswith("laplace"):
                ret = (
                    torch.distributions.Laplace(0, 1)
                    .sample((n_samples, 1 if noise_type.endswith("repeated") else dim))
                    .to(self.device)
                )
            else:
                raise ValueError(f"Invalid noise type! ({noise_type})")
        if noise_type.endswith("repeated"):
            # repeat ret along the second axis for "dim" times
            ret = ret.repeat((1, dim))
        return ret

    def _train_diffeomorphisms_loss(
        self,
        diffeomorphism: Diffeomorphism,
        domain: torch.Tensor,
        condition_number_neighbourhood_threshold: float,
    ):
        """
        Train to enforce a condition number of '1' for the diffeomporphism. Check the docstring of the class for more details.

        Args:
            diffeomorphism (Diffeomorphism): The diffeomorphism to train
            domain (torch.Tensor): A set of samples in the domain of the diffeomorphism to adjust the condition number.
            condition_number_neighbourhood_threshold (float): The threshold used for neighbourhood in
                the loss, this is used to fill out msk_ij that was explained in the class docstring.
        Returns:
            The loss value
        """
        domain_pairwise_distance = torch.cdist(domain, domain)
        domain_mask = (domain_pairwise_distance < condition_number_neighbourhood_threshold).int()
        projection = diffeomorphism.apply_transform(domain)
        projection_pairwise_distance = torch.cdist(projection, projection)
        projection_mask = (
            projection_pairwise_distance < condition_number_neighbourhood_threshold
        ).int()
        kappa = (projection_pairwise_distance + 1e-3) / (domain_pairwise_distance + 1e-3)
        mask = domain_mask * projection_mask
        loss = torch.sum(
            mask
            * (
                kappa + 1 / kappa - 2
            )  # a loss function that is zero iff the function has condition number 1
        ) / (torch.sum(mask) + 1e-3)
        return loss

    def _get_affine_projection(self, affine_projection_type: str):

        # linear transform to higher dimension
        if affine_projection_type.startswith("random"):

            def proj(x, i):
                linear_projection = self.proj_matrix[i]
                # compute the SVD of the linear projection
                U, S, V = torch.svd(linear_projection)
                if affine_projection_type == "random-rotation":
                    S = torch.eye(self.manifold_dims[i], device=self.device)
                else:
                    S = torch.diag(S)
                return (U @ S @ V.T @ x.T).T

        elif affine_projection_type == "repeat":
            # repeat the columns of 'x' to make it 'ambient_dim' dimensional but if it is not divisible by 'mdim' then
            # the ramaining columns are repeated from the first column
            proj = lambda x, i: torch.tile(
                x,
                (
                    1,
                    (self.ambient_dim + self.manifold_dims[i] - 1) // self.manifold_dims[i],
                ),
            )[:, : self.ambient_dim]
        elif affine_projection_type == "zero-pad":
            # zero-pad the columns of 'x' to make it 'ambient_dim' dimensional

            proj = lambda x, i: (
                torch.cat(
                    (
                        x,
                        torch.zeros(
                            x.shape[0],
                            self.ambient_dim - self.manifold_dims[i],
                            device=self.device,
                        ),
                    ),
                    dim=1,
                )
                if self.manifold_dims[i] < self.ambient_dim
                else x
            )
        else:
            raise Exception(
                f"Invalid value for 'affine_projection_type'! ({affine_projection_type})"
            )
        return proj

    def _train_diffeomorphism(
        self,
        i: int,
        proj: Callable[[torch.Tensor, int], torch.Tensor],
        n_calibration: int,
        mdim: int,
        lr: float,
        n_iter_calibration: int,
        verbose: int,
        cnt_training: int,
        to_train_or_not_to_train: List[bool],
        bin_step: int,
        condition_number_neighbourhood_threshold: float,
        bin_search_n_iter,
    ):
        # Make the training process non-stochastic for better results in the ternery search
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)

            loss_history = []
            # Train the diffeomorphism to be stable around uniform distribution
            trainer_iterator = range(n_iter_calibration)
            if verbose > 0:
                trainer_iterator = tqdm(
                    trainer_iterator,
                    desc=f"Training diffeomorphism [{cnt_training}/{sum(to_train_or_not_to_train)}] bin search step [{bin_step}/{bin_search_n_iter}] with lr={lr} ...",
                )
            optim = torch.optim.Adam(
                self.diffeo[i].parameters(),
                lr=lr,
            )
            self.diffeo[i].train()
            self.diffeo[i].to(self.device)

            for epoch in trainer_iterator:
                # a flag for whether the training has diverged!

                with torch.no_grad():
                    latent = self._sample_latent(
                        n_calibration,
                        mdim,
                        seed=epoch * len(self.manifold_dims) + i,
                    )
                    ambient = proj(latent, i)
                    noised_out_ambient = ambient + torch.rand_like(ambient) * 2 - 1

                loss = self._train_diffeomorphisms_loss(
                    self.diffeo[i],
                    noised_out_ambient,
                    condition_number_neighbourhood_threshold,
                )
                if verbose > 0:
                    trainer_iterator.set_postfix({"loss": loss.item()})
                loss_history.append(loss.item())
                optim.zero_grad()
                loss.backward()
                optim.step()

        return (
            sum(loss_history[-ManifoldMixture.LOSS_AVG_WINDOW_SIZE :])
            / ManifoldMixture.LOSS_AVG_WINDOW_SIZE
        )

    def _init_projections(
        self,
        affine_projection_type: str,
        diffeomorphism_instantiator: (
            List[Tuple[DiffeomorphicConstructor | None, DiffeomorphicConstructor | None]] | None
        ),
        adjust_condition_number: bool,
        condition_number_neighbourhood_threshold: float,
        n_calibration: int,
        n_iter_calibration: int,
        lr_calibration: float | None,
        verbose: int,
    ):
        """
        Setup num_mixtures different projectors. Each projector has two components:

        1. A affine transformation that projects the distribution to a higher dimensional space.
        2. A diffeomorphism that projects the distribution in the ambient space to a more complex one,
            while preserving the intrinsic dimension of the manifold.

        The first and third element are only active if their corresponding boolean flags are set to True.

        The affine transformation can be of four types:
        1. Random: A random linear transformation is applied to the distribution.
        2. Random-rotation: A random linear transformation is sampled but it goes through a QR decomposition to ensure that it is orthogonal.
        2. Repeat: The columns of the distribution are repeated to make it higher dimensional.
        3. Zero-pad: The columns of the distribution are zero-padded to make it higher dimensional.

        Finally, after the transformations are applied, the projector is normalized to ensure that the marginals produce numerically stable results.

        **More details**:
            If `adjust_condition_number=True` a post-hoc training is done on the diffeomorphism to control their condition number.
            All of the following is an experimental feature and it is recommended to instead use simple diffeomorphisms that are stable and have a controlled condition number.
            But if you want to use this feature, then the diffeomorphisms are trained to be stable around the generated samples.
            This is done by optimizing the condition number loss (documented in `_train_diffeomorphisms_loss` method) for a set of random samples
            in the domain of the diffeomorphism. The training is done using Adam optimizer with a learning rate of `lr_calibration` or a ternery search
            over the learning rate to find the best learning rate that minimizes the condition number loss. This is to ensure that lr_calibration is not
            required to be set manually. The training is done for `n_iter_calibration` iterations.
        """

        # handle typing and set the diffeomorphism instantiators
        if not diffeomorphism_instantiator:
            diffeomorphism_instantiator = [None for _ in range(self.n_mixtures)]
        self.diffeomorphism_instantiator = diffeomorphism_instantiator

        self.projectors = []  # this would be filled up by actual functions

        # Instantiate and parameterize the diffemorphisms and set the linear projection matrices
        self.diffeo = []
        self.proj_matrix = []
        for i, mdim in enumerate(self.manifold_dims):
            if self.diffeomorphism_instantiator[i] is not None:
                diffeomorphism = self.diffeomorphism_instantiator[i](self.ambient_dim).to(
                    self.device
                )
                self.diffeo.append(diffeomorphism)
            else:
                # create a dummy nn.Module that does nothing
                self.diffeo.append(None)
            self.proj_matrix.append(torch.randn((self.ambient_dim, mdim), device=self.device))

        # a boolean vector indicating which diffeomorphisms we are able to train
        # ones that are None or are not parameterized using a torch module
        # are not trainable
        to_train_or_not_to_train = [
            (
                self.diffeo[i] is not None
                and adjust_condition_number
                and any(True for _ in self.diffeo[i].parameters())
            )
            for i in range(self.n_mixtures)
        ]

        # Define a function that takes a data and a mode index and returns the diffeomorphism applied to the data
        diffeo_fn = lambda x, i: (
            self.diffeo[i].apply_transform(x)
            if self.diffeomorphism_instantiator[i] is not None
            else x
        )

        # Define the projection function
        proj = self._get_affine_projection(affine_projection_type)

        # Train everything and calbrate
        self.calibrate_mean = []
        self.calibrate_std = []
        cnt_training = 0
        for i, mdim in enumerate(self.manifold_dims):
            assert (
                mdim <= self.ambient_dim
            ), f"Manifold dimension should be less than or equal to the ambient dimension! (ambient dim idx = {i} val = {mdim})"
            # sample a set of points to get empirical mean and std of marginals

            if to_train_or_not_to_train[i]:

                cnt_training += 1

                # store the initial state of the diffeomorphism
                initial_state = copy.deepcopy(self.diffeo[i].state_dict())

                # Perform a ternery search on the best learning curve that ends up optimizing the condition number
                if lr_calibration is None:
                    lr_l = 1e-8
                    lr_r = 1e-1
                    loss_1 = None
                    loss_2 = None
                    for bin_step in range(ManifoldMixture.BIN_SEARCH_LR_N_ITER):
                        lr_1 = (2 * lr_l + lr_r) / 3
                        lr_2 = (lr_l + 2 * lr_r) / 3
                        self.diffeo[i].load_state_dict(initial_state)
                        loss_1 = self._train_diffeomorphism(
                            i=i,
                            proj=proj,
                            n_calibration=n_calibration,
                            mdim=mdim,
                            lr=lr_1,
                            n_iter_calibration=n_iter_calibration,
                            verbose=verbose,
                            cnt_training=cnt_training,
                            to_train_or_not_to_train=to_train_or_not_to_train,
                            bin_step=2 * bin_step + 1,
                            condition_number_neighbourhood_threshold=condition_number_neighbourhood_threshold,
                            bin_search_n_iter=2 * ManifoldMixture.BIN_SEARCH_LR_N_ITER + 1,
                        )
                        # check if loss is nan or is large
                        if loss_1 != loss_1 or loss_1 > 10:
                            lr_r = lr_1
                            continue
                        self.diffeo[i].load_state_dict(initial_state)
                        loss_2 = self._train_diffeomorphism(
                            i=i,
                            proj=proj,
                            n_calibration=n_calibration,
                            mdim=mdim,
                            lr=lr_2,
                            n_iter_calibration=n_iter_calibration,
                            verbose=verbose,
                            cnt_training=cnt_training,
                            to_train_or_not_to_train=to_train_or_not_to_train,
                            bin_step=2 * bin_step + 2,
                            condition_number_neighbourhood_threshold=condition_number_neighbourhood_threshold,
                            bin_search_n_iter=2 * ManifoldMixture.BIN_SEARCH_LR_N_ITER + 1,
                        )
                        if loss_1 < loss_2:
                            lr_r = lr_2
                        else:
                            lr_l = lr_1
                    if loss_2 is None or loss_1 < loss_2:
                        lr = lr_1
                    else:
                        lr = lr_2
                else:
                    lr = lr_calibration

                # train the diffeomorphism
                self.diffeo[i].load_state_dict(initial_state)
                best_loss = self._train_diffeomorphism(
                    i=i,
                    proj=proj,
                    n_calibration=n_calibration,
                    mdim=mdim,
                    lr=lr,
                    n_iter_calibration=n_iter_calibration,
                    verbose=verbose,
                    cnt_training=cnt_training,
                    to_train_or_not_to_train=to_train_or_not_to_train,
                    bin_step=2 * ManifoldMixture.BIN_SEARCH_LR_N_ITER + 1,
                    condition_number_neighbourhood_threshold=condition_number_neighbourhood_threshold,
                    bin_search_n_iter=2 * ManifoldMixture.BIN_SEARCH_LR_N_ITER + 1,
                )

                if best_loss > 10:
                    print(
                        f"[Warning!] Can't get small condition number for the diffeomorphism {i} make the diffeomorphisms less expressive! (loss = {best_loss})"
                    )

            # Now we will separate out the mixtures by first standardizing the projections
            # using a set of calibration samples and then add the modes to the standardized
            # projections. This will ensure minimal mixing between the modes.
            with torch.no_grad():
                latent = self._sample_latent(n_calibration, mdim, seed=i)
                calibrate_x: torch.Tensor = diffeo_fn(proj(latent, i), i)

            calibrate_mean = calibrate_x.mean(dim=0, keepdim=True)
            calibrate_std = calibrate_x.std(dim=0, keepdim=True)
            calibrate_std[calibrate_std < 1.0] = 1.0
            self.calibrate_mean.append(calibrate_mean)
            self.calibrate_std.append(calibrate_std)

            self.projectors.append(
                lambda x, i=i: (diffeo_fn(proj(x, i), i) - self.calibrate_mean[i])
                / self.calibrate_std[i]
                + self.modes[i]
            )

    def __init__(
        self,
        manifold_dims: List[int] | int,
        ambient_dim: int,
        diffeomorphism_instantiator: List[DiffeomorphicConstructor | None] | None = None,
        affine_projection_type: Literal[
            "random", "repeat", "zero-pad", "random-rotation"
        ] = "random-rotation",
        mixture_probs: List[float] | None = None,  # when set to None it is uniform
        sample_distr: Literal["normal", "uniform", "laplace"] = "uniform",
        distance_between_modes: float = 1.0,
        seed: int = 42,
        device: torch.device = torch.device("cpu"),
        adjust_condition_number: bool = False,
        condition_number_neighbourhood_threshold: float | None = None,
        n_calibration: int = 128,
        n_iter_calibration: int = 10,
        lr_calibration: float | None = None,
        verbose: int = 1,
    ):
        """
        Args:
            manifold_dims (List[int] | int): The intrinsic dimension
            ambient_dim (int): The dimension of the ambient space
            diffeomorphism_instantiator:
                A list of partials that take in the dimension of the manifold and return a diffeomorphism.
            project_in_latent_space (bool): Whether to project in the latent space
                using the diffeomorphism_instantiator or not.
            project_in_ambient_space (bool): Whether to project in the ambient space
                using the diffeomorphism_instantiator or not.
            affine_projection_type (Literal["random", "repeat"]): The type of affine
                projection to use, either a random linear transformation or repeating
                the columns of the distribution.
            mixture_probs (List[float] | None): The mixture probabilities of the modes.
            sample_distr (Literal["normal", "uniform", "laplace"]): The distribution
                to sample in the latent space.
            distance_between_modes (float): The (lower bound) of distance between the modes,
                set it to larger for more separation.
            seed (int): The seed for reproducibility.

            condition_number_neighbourhood_threshold (float): The threshold used for neighbourhood in
                the loss.
            n_calibration: The number of samples to calibrate the marginals of the projectors.
            n_iter_calibration: The number of iterations to train the diffeomorphisms.
            lr_calibration: The learning rate for adjusting the diffeomorphisms.

            verbose (int): The verbosity level.
        """
        # set the device
        self.device = device
        if self.device != torch.device("cpu") and seed is not None:
            print(
                "[Warning!] When constructing a mixture of manifolds, setting seed for reproducibility on GPU is not recommended, change the device to CPU or remove the seed argument in the seed constructor!"
            )
        if adjust_condition_number:
            print(
                "[Warning] Adjusting the condition number of diffeomorphisms is an experimental feature, instead, we recommend using simple diffeomorphisms that are stable and have a controlled condition number!"
            )
        self.verbose = verbose
        if verbose > 0 and adjust_condition_number:
            print("Initializing ManifoldMixture ...")
            print("[set verbose=0 in the constructor to suppress]")
        self.seed = seed if seed is not None else torch.randint(0, 100000, (1,)).item()
        self.ambient_dim = ambient_dim
        if isinstance(manifold_dims, int):
            manifold_dims = [manifold_dims]
        self.n_mixtures = len(manifold_dims)
        self.manifold_dims = manifold_dims
        self.sample_distr = sample_distr

        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            self._init_mixture_probs(mixture_probs)
            self._init_modes(ambient_dim, distance_between_modes)

        if condition_number_neighbourhood_threshold is None:
            condition_number_neighbourhood_threshold = 0.1
        with torch.random.fork_rng():
            torch.manual_seed(self.seed + 1)
            self._init_projections(
                affine_projection_type=affine_projection_type,
                diffeomorphism_instantiator=diffeomorphism_instantiator,
                adjust_condition_number=adjust_condition_number,
                condition_number_neighbourhood_threshold=condition_number_neighbourhood_threshold,
                n_iter_calibration=n_iter_calibration,
                n_calibration=n_calibration,
                verbose=verbose,
                lr_calibration=lr_calibration,
            )

    @torch.no_grad()
    def sample(
        self,
        sample_shape: int | Tuple[int],
        return_dict: bool = False,
        chunk_size: int = 128,
        seed: int | None = None,
    ):
        """

        Args:
            sample_shape (int | Tuple[int]): It can either be the number of samples or it can be a tuple (n, ambient_dim)
            return_dict (bool, optional): When set to True, a dictionary of {'samples', 'idx', 'lid'} is returned, where
                the first is the actual data of size (n, ambient_dim), the second is a vector (n, ) that indicates
                what mode does every datapoint lie in, `lid` is a vector of size (n, ) which is the ground truth intrinsic
                dimension. This is added because it is mainly a benchmark for LID estimation.
            chunk_size (int, optional):
                When specified, the sampling is done batch-by-batch. This is useful to prevent memory overflow.
                in GPU or CPU. Defaults to 128.
            seed (int | None, optional): For reproducibility. Defaults to None.

        Returns:
            torch.Tensor of shape (n, ambient_dim) or a dictionary of {'samples', 'idx', 'lid'}.
        """
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape, self.ambient_dim)
        # check shape to be N x ambient_dim
        assert len(sample_shape) == 1 or (
            len(sample_shape) == 2 and sample_shape[1] == self.ambient_dim
        ), "Sample shape should be N x ambient_dim"

        all_samples_list = []
        all_lid_list = []
        all_idx_list = []
        seed = seed or torch.randint(0, 100000, (1,)).item()
        my_iterator = range(0, sample_shape[0], chunk_size)
        if self.verbose > 1:
            my_iterator = tqdm(my_iterator)
        for i in my_iterator:
            sz = min(chunk_size, sample_shape[0] - i)

            # sample a bunch of uniform random numbers and use them
            # to find the index of the mode to use for each sample
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                u = torch.rand(sz).to(self.device)
            mode_idx = torch.searchsorted(self.mixture_probs_cdf, u)

            samples = torch.zeros((sz, self.ambient_dim), device=self.device)
            for j, mdim in enumerate(self.manifold_dims):
                idx = mode_idx == j
                if not idx.any():
                    continue
                latent = self._sample_latent(
                    idx.sum(),
                    mdim,
                    seed=(i * len(self.manifold_dims) + j + seed if seed is not None else None),
                ).to(self.device)
                x = self.projectors[j](latent)
                samples[idx] = x
            all_samples_list.append(samples.cpu())
            all_lid_list.append(
                torch.IntTensor(self.manifold_dims, device=torch.device("cpu"))[mode_idx.cpu()]
            )
            all_idx_list.append(mode_idx.cpu())

        all_samples = torch.cat(all_samples_list, dim=0)
        all_lid = torch.cat(all_lid_list, dim=0)
        all_idx = torch.cat(all_idx_list, dim=0)

        if return_dict:
            return {
                "samples": all_samples,
                "lid": all_lid,
                "idx": all_idx,
            }
        return all_samples


class SquigglyManifoldMixture(ManifoldMixture):
    """
    This is a manifold mixture which is hard but fair for LID estimation!

    The idea is similar to the Mixture of manifolds but the diffeomorphisms are set in a specific
    way to ensure that the condition number is controlled. To do so, we perform the following
    diffeomorphism:

    1. for a given 'x', randomly sample a subset of indices and replace them with x + 0.5 sin(freq * x) / freq
    2. randomly sample a subset of indices and replace them with -x
    3. perform a random rotation using a random orthonormal matrix of size d by d
    4. repeat the steps 1 to 3 and chain all of the transforms together for 'n_transforms' times.

    All the orthogonal rotations have a condition number of '1' and x + 0.5 sin(freq * x) / freq also has a condition
    number close to 1 when frequency is relatively small. Thus, chaining them together will ensure that the condition
    number is also controlled.

    Note that by increasing 'n_transforms' and 'frequency', the manifold becomes more complex and harder to estimate
    LID for. Thus, it provides a good benchmark for LID estimation.
    """

    def __init__(
        self,
        manifold_dims: List[int] | int,
        ambient_dim: int,
        mixture_probs: List[float] | None = None,  # when set to None it is uniform
        sample_distr: Literal["normal", "uniform", "laplace"] = "uniform",
        distance_between_modes: float = 1.0,
        seed: int = 42,
        device: torch.device = torch.device("cpu"),
        frequency: float = 10.0,
        kappa_control: float = 1e-4,
        n_transforms: int = 5,
    ):
        """
        Please check the docstring of the parent class for more details.
        Here, we need not define a diffeomorphism instantiator as it is already defined in the class.

        Args:
            n_transforms (int, optional): The number of repetitions. Defaults to 5, make it larger to make it harder.
            frequency (float): Increasing the frequency would increase the fluctuations, leading to harder LID estimation.
            kappa_control (float):
                A number in (0, 1]. When it is set to (almost) zero, there is no control over the condition number and
                when it is 1 the condition number stays 1.
        """
        super().__init__(
            manifold_dims=manifold_dims,
            ambient_dim=ambient_dim,
            diffeomorphism_instantiator=[
                functools.partial(
                    Sinusoidal,
                    repeat=n_transforms,
                    frequency=frequency,
                    kappa_control=kappa_control,
                    seed=seed + i,
                )
                for i in range(len(manifold_dims))
            ],
            affine_projection_type="random-rotation",
            mixture_probs=mixture_probs,
            sample_distr=sample_distr,
            distance_between_modes=distance_between_modes,
            seed=seed,
            device=device,
            adjust_condition_number=False,
            verbose=0,
        )


class AffineManifoldMixture(ManifoldMixture):
    """
    This is a manifold mixture that uses affine transformations to project the distribution
    to a higher dimensional space.
    """

    def __init__(
        self,
        manifold_dims: List[int] | int,
        ambient_dim: int,
        affine_projection_type: Literal["random", "repeat", "zero-pad", "random-rotation"],
        mixture_probs: List[float] | None = None,  # when set to None it is uniform
        sample_distr: Literal["normal", "uniform", "laplace"] = "uniform",
        distance_between_modes: float = 1.0,
        seed: int = 42,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            manifold_dims=manifold_dims,
            ambient_dim=ambient_dim,
            diffeomorphism_instantiator=None,
            affine_projection_type=affine_projection_type,
            mixture_probs=mixture_probs,
            sample_distr=sample_distr,
            distance_between_modes=distance_between_modes,
            seed=seed,
            device=device,
            adjust_condition_number=False,
            verbose=0,
        )
