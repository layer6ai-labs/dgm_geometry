"""
A lightweight trainer, well-suited for experiments in the notebooks.

Instead of passing the lightning modules to a lightning Trainer, you can
use this trainer that would side-step all the overhead of lightning. However,
it will not support any of the advanced features such as callbacks.
"""

import os

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm


class LightweightTrainer:

    def __init__(
        self,
        max_epochs: int,
        default_root_dir: (
            str | None
        ) = None,  # The default root directory for saving the checkpoints
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_checkpoints: int = 5,
    ) -> None:
        """
        This is a lightweight trainer to replace lightning trainers when we don't need them.

        Args:
            max_epochs: The maximum number of epochs to train
            default_root_dir: The root directory which is used to store all the checkpoints
            device: The torch device being used for training
            num_checkpoints:
                The number of checkpoints to store in the training process. Typically, these are
                snapshots of the model in regular time intervals.
        """
        self.default_root_dir = default_root_dir
        self.max_epochs = max_epochs
        self.device = device
        self.num_checkpoints = num_checkpoints

    def _remove_extra_checkpoints(self, ckpt_path: str):
        """
        Remove the extra checkpoints from the directory

        Args:
            ckpt_path: The path to the directory containing the checkpoints
        """
        if self.num_checkpoints is not None:
            # Iterate over ckpt to see the last epoch
            ckpt_files = []
            for file in os.listdir(ckpt_path):
                # if it starts with "epoch_" and ends with ".ckpt"
                if file.startswith("epoch_") and file.endswith(".ckpt"):
                    ckpt_files.append(file)
            ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split("_")[1]))
            for file in ckpt_files[: -self.num_checkpoints]:
                os.remove(os.path.join(ckpt_path, file))

    def _load_last_checkpoint(self, ckpt_path: str, model: torch.nn.Module):
        """
        Load the last checkpoint from the directory

        Args:
            ckpt_path: The path to the directory containing the checkpoints
            model: The model to load the checkpoint into
        """
        # Iterate over ckpt to see the last epoch
        mx_epoch = -1
        ckpt_file = None
        for file in os.listdir(ckpt_path):
            # if it starts with "epoch_" and ends with ".ckpt"
            if file.startswith("epoch_") and file.endswith(".ckpt"):
                epoch_cnt = int(file.split("_")[1])
                if epoch_cnt > mx_epoch:
                    mx_epoch = epoch_cnt
                    ckpt_file = file
        if ckpt_file is not None:
            ckpt_file = os.path.join(ckpt_path, ckpt_file)
            model.load_state_dict(torch.load(ckpt_file))
        return mx_epoch

    def fit(
        self,
        model: torch.nn.Module,
        train_dataloader: TorchDataLoader,
        val_dataloader: TorchDataLoader | None = None,
        ckpt_path: str | None = None,
    ):
        """
        Args:
            model: The model to train
            train_dataloader: The training dataloader
            val_dataloader: The validation dataloader
            ckpt_path: The path to save the checkpoints
        """
        # check if model has the required methods
        assert hasattr(
            model, "configure_optimizers"
        ), "Model should have `configure_optimizers` method"
        assert hasattr(model, "loss"), "Model should have `loss` method"
        assert hasattr(model, "unpack_batch"), "Model should have `unpack_batch` method"
        assert hasattr(model, "perturb_batch"), "Model should have `perturb_batch` method"

        # load checkpoint if available
        current_epoch = 0
        modulo = 1
        if ckpt_path is not None:
            if self.default_root_dir is not None:
                ckpt_path = os.path.join(self.default_root_dir, ckpt_path)
            os.makedirs(ckpt_path, exist_ok=True)
            current_epoch = self._load_last_checkpoint(ckpt_path, model)
            new_epoch_count = self.max_epochs - current_epoch
            modulo = max(1, new_epoch_count // self.num_checkpoints)

        # extract optimizers and schedulers
        ret = model.configure_optimizers()
        model = model.to(self.device)
        if isinstance(ret, tuple):
            assert len(ret) == 2, f"Expected tuple of length 2, got {len(ret)}"
            optim = ret[0]
            if isinstance(optim, list):
                assert len(optim) == 1, f"Expected list of length 1, got {len(optim)}"
                optim: torch.optim.Optimizer = optim[0]
            scheduler = ret[1]
            if isinstance(scheduler, list):
                assert len(scheduler) == 1, f"Expected list of length 1, got {len(scheduler)}"
                scheduler: (
                    torch.optim.lr_scheduler.LRScheduler
                    | torch.optim.lr_scheduler.ReduceLROnPlateau
                ) = scheduler[0]
        else:
            optim = ret
            scheduler = None
        assert isinstance(
            optim, torch.optim.Optimizer
        ), f"Expected torch.optim.Optimizer, got {type(optim)}"
        assert (
            scheduler is None
            or isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler)
            or isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        ), f"Expected None or torch.optim.lr_scheduler.LRScheduler, got {type(scheduler)}"

        # main training loop
        training_iterator = tqdm(
            range(current_epoch, self.max_epochs),
            desc="Training",
            total=self.max_epochs,
            initial=current_epoch,
        )
        for epoch_cnt in training_iterator:
            loss_history = []
            batch_idx = 0
            model.train()
            for batch in train_dataloader:
                training_iterator.set_description(
                    f"Training epochs [Batch {batch_idx+1}/{len(train_dataloader)}]"
                )
                optim.zero_grad()
                batch = model.unpack_batch(batch)
                batch: torch.Tensor = model.perturb_batch(batch)
                batch = batch.to(self.device)
                loss: torch.Tensor = model.loss(batch)
                loss.backward()
                loss_history.append(loss.mean().item())
                optim.step()
                batch_idx += 1
            avg_train_loss = sum(loss_history) / len(loss_history)
            training_iterator.set_postfix({"loss": avg_train_loss})

            # apply scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_train_loss)
                else:
                    scheduler.step()

            # validation
            model.eval()
            with torch.no_grad():
                if val_dataloader is not None:
                    loss_history = []
                    batch_idx = 0
                    for batch in val_dataloader:
                        training_iterator.set_description(
                            f"Validation epochs [Batch {batch_idx+1}/{len(train_dataloader)}]"
                        )
                        batch = model.unpack_batch(batch)
                        batch: torch.Tensor = model.perturb_batch(batch)
                        batch = batch.to(self.device)
                        loss: torch.Tensor = model.loss(batch)
                        loss_history.append(loss.mean().item())
                        batch_idx += 1
                    avg_val_loss = sum(loss_history) / len(loss_history)
                    training_iterator.set_postfix(
                        {"val_loss": avg_val_loss, "loss": avg_train_loss}
                    )

            # Save the model checkpoint
            if ckpt_path is not None and (epoch_cnt - current_epoch + modulo - 1) % modulo == 0:
                avg_train_loss_rounded = round(avg_train_loss, 4)
                path = os.path.join(
                    ckpt_path, f"epoch_{epoch_cnt}_loss={avg_train_loss_rounded}.ckpt"
                )
                torch.save(
                    model.state_dict(),
                    path,
                )

        if ckpt_path is not None:
            self._remove_extra_checkpoints(ckpt_path)
