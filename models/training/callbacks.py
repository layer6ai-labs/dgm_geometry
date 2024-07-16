import mlflow
import torch
import torch.utils
import torchvision.transforms.functional as TVF
import umap
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid
from tqdm import tqdm

from models.training.lightning_dgm import LightningDGM
from models.training.lightning_ensemble import LightningEnsemble
from visualization import visualize_umap_clusters
from visualization.pretty import ColorTheme


def _parse_sampling_kwargs(sampling_kwargs) -> dict:
    """Turns an Omegaconf ListConfig or DictConfig into a normal dictionary"""

    if isinstance(sampling_kwargs, list):
        ret = []
        for kwargs in sampling_kwargs:
            ret.append(_parse_sampling_kwargs(kwargs))
        return ret

    parsed_sampling_kwargs = {}
    if sampling_kwargs is not None:
        for key in sampling_kwargs.keys():
            if key == "sample_shape":
                new_value = list([x for x in sampling_kwargs[key]])
                parsed_sampling_kwargs[key] = tuple(new_value)
            else:
                parsed_sampling_kwargs[key] = sampling_kwargs[key]
    return parsed_sampling_kwargs


class SampleGrid(Callback):
    """Sample a grid of images to MLflow"""

    def __init__(
        self,
        path_fstr="sample_grids/epoch={epoch:04d}-step={step:07d}.png",
        sample_every_n_epoch=None,
        sample_every_n_step=None,
        transform=None,
        sample_kwargs=None,
        grid_kwargs=None,
        seed=0,
    ):
        self.path_fstr = path_fstr
        self.sample_every_n_epoch = sample_every_n_epoch
        self.sample_every_n_step = sample_every_n_step
        self.transform = transform
        self.seed = seed

        assert sample_every_n_step or sample_every_n_epoch

        sample_kwargs = sample_kwargs or {}
        if "num" not in sample_kwargs:
            sample_kwargs["num"] = 64
        self.sample_kwargs = _parse_sampling_kwargs(sample_kwargs)

        if grid_kwargs is not None:
            self.grid_kwargs = grid_kwargs
        else:
            self.grid_kwargs = {}

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        assert not isinstance(pl_module, LightningEnsemble), "Ensemble not supported!"

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        if self.sample_every_n_step and pl_module.global_step % self.sample_every_n_step == 0:
            self._save_grid_to_mlflow(pl_module)

    def on_train_epoch_end(self, trainer, pl_module: LightningDGM):
        if self.sample_every_n_epoch and pl_module.current_epoch % self.sample_every_n_epoch == 0:
            self._save_grid_to_mlflow(pl_module)

    def _save_grid_to_mlflow(self, pl_module: LightningDGM):
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            sample = pl_module.sample(**self.sample_kwargs)

        grid_tensor = make_grid(sample, **self.grid_kwargs)
        grid_pil = TVF.to_pil_image(grid_tensor)

        out_path = self.path_fstr.format(epoch=pl_module.current_epoch, step=pl_module.global_step)
        mlflow.log_image(grid_pil, out_path)


class MlFlowLogMetrics(Callback):
    """Push any metrics recorded with self.log to MLflow"""

    def on_train_epoch_end(self, trainer, pl_module):
        mlflow.log_metrics(
            {key: float(val) for key, val in trainer.callback_metrics.items()},
            step=pl_module.global_step,
        )


class UmapGeneratorEval(Callback):
    """generate samples and contrast them with the samples in the dataset"""

    def __init__(
        self,
        frequency: int = 1,
        limit_count: int = 1000,
        path_fstr: str = "sample_embeddings/real_vs_generated_epoch={epoch_num:04d}",
        sampling_kwargs: dict | list | None = None,
        verbose: bool = True,
        use_same_reducer: bool = False,
    ):
        """
        Get a frequency, and every frequency epochs, sample from the generator and
        contrast with the dataset and perform UMAP on the samples, finally, this
        Umap embedding is logged onto MLflow, showing the quality of sample generation.
        All of these samples are stored in 'umap/real_vs_generated_samples_{cnt}.png'

        Args:
            dataset (torch.utils.data.Dataset):
                A dataset that represents the real data
            frequency (int, optional):
                The frequency at which the image is plotted.
            limit_count (int, optional):
                The maximum number of points to be picked for plotting.
        """
        self.frequency = frequency
        self.rem = self.frequency
        self.limit_count = limit_count

        self.sampling_kwargs = _parse_sampling_kwargs(sampling_kwargs or {})

        self.path_fstr_ensemble = path_fstr + "_model={idx:01d}.png"
        self.path_fstr = path_fstr + ".png"

        self._first_time = True
        self.verbose = verbose
        self.use_same_reducer = use_same_reducer

    def _store_candidate_data(
        self,
        pl_module: LightningDGM,
        batch: torch.Tensor,
        model_idx: int = 0,
    ):
        batch = pl_module.unpack_batch(batch)
        batch = pl_module.perturb_batch(batch)

        if hasattr(self, "training_data_tensor") and model_idx in self.training_data_tensor:
            if len(self.training_data_tensor[model_idx]) > self.limit_count:
                self.training_data_tensor[model_idx] = self.training_data_tensor[model_idx][
                    : self.limit_count
                ]
            elif len(self.training_data_tensor[model_idx]) == self.limit_count:
                return
            else:
                self.training_data_tensor[model_idx] = torch.cat(
                    [self.training_data_tensor[model_idx], batch], dim=0
                )
        else:
            if not hasattr(self, "training_data_tensor"):
                self.training_data_tensor: dict = {}
            self.training_data_tensor[model_idx] = batch

    def on_train_batch_end(
        self,
        trainer,
        pl_module: LightningDGM | LightningEnsemble,
        outputs,
        batch,
        batch_idx,
    ):
        """gathers all the training data in self.training_data_tensor to initialize the reducer"""
        # skip if it is not the first time running
        if not self._first_time:
            return

        if isinstance(pl_module, LightningEnsemble):
            for i in range(len(pl_module.lightning_dgms)):
                self._store_candidate_data(pl_module.lightning_dgms[i], batch, model_idx=i)
        elif isinstance(pl_module, LightningDGM):
            self._store_candidate_data(pl_module, batch)
        else:
            assert False, "The model should be either LightningDGM or LightningEnsemble"

    def _init_reducer(self):
        # initialize a UMAP embedder on all the data
        self.reducers = {}
        rng = self.training_data_tensor.keys()
        if self.verbose:
            rng = tqdm(rng, desc="[UMap Callback] Initializing UMAP reducer ...")
        lst_reducer = None
        for key in rng:
            flattened_data = self.training_data_tensor[key].flatten(start_dim=1)
            if self.use_same_reducer and lst_reducer is not None:
                self.reducers[key] = lst_reducer
            else:
                self.reducers[key] = umap.UMAP()
                self.reducers[key].fit(flattened_data.cpu().numpy())
            lst_reducer = self.reducers[key]

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningDGM | LightningEnsemble
    ) -> None:
        # if it is the first time reaching an epoch end then initialize the reducer
        if self._first_time:
            self._init_reducer()
            self._first_time = False  # set _first_time flag to False

        # only log whenever the frequency is reached
        self.rem -= 1
        if self.rem > 0:
            return
        self.rem = self.frequency

        if self.verbose:
            print(f"[UMap Callback] Getting samples ...")
        generated_samples = pl_module.sample(num=self.limit_count, **self.sampling_kwargs)
        if isinstance(pl_module, LightningDGM):
            generated_samples = [generated_samples]

        for key in self.training_data_tensor.keys():
            flattenned_gen = generated_samples[key].flatten(start_dim=1)
            flattenned_x = self.training_data_tensor[key].flatten(start_dim=1)
            # automatically set alpha according to the size of flattened_x
            if flattenned_x.shape[0] > 1000:
                alpha = 0.1
            else:
                alpha = 0.5 - 0.4 * flattenned_x.shape[0] / 1000

            pil_img = visualize_umap_clusters(
                data=[flattenned_x, flattenned_gen],
                labels=["real", "generated"],
                colors=[ColorTheme.BLUE_SECOND.value, ColorTheme.GOLD.value],
                title="Real vs Generated samples",
                alpha=alpha,
                return_img=True,
                reducer=self.reducers[key],
            )

            if isinstance(pl_module, LightningEnsemble):
                mlflow.log_image(
                    pil_img,
                    self.path_fstr_ensemble.format(epoch_num=pl_module.current_epoch, idx=key),
                )
            else:
                mlflow.log_image(pil_img, self.path_fstr.format(epoch_num=pl_module.current_epoch))
