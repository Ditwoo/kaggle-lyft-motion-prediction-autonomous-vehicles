import os
import time
import shutil
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader, RandomSampler

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.evaluation import create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.rasterization import build_rasterizer

from src.batteries import (
    seed_all,
    t2d,
    zero_grad,
    CheckpointManager,
    TensorboardLogger,
    make_checkpoint,
    load_checkpoint,
)
from src.batteries.progress import tqdm
from src.models import ModelWithConfidence
from src.models.resnet3d import resnet18_3d
from src.criterion import neg_multi_log_likelihood_batch


torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


DEBUG = int(os.environ.get("DEBUG", -1))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "16"))
os.environ["L5KIT_DATA_FOLDER"] = "./data"

cfg = {
    "format_version": 4,
    "model_params": {
        "history_num_frames": 20,
        "history_step_size": 1,
        "history_delta_time": 0.1,
        "future_num_frames": 50,
        "future_step_size": 1,
        "future_delta_time": 0.1,
    },
    "raster_params": {
        "raster_size": [224, 224],
        "pixel_size": [0.5, 0.5],
        "ego_center": [0.25, 0.5],
        "map_type": "py_semantic",
        "satellite_map_key": "aerial_map/aerial_map.png",
        "semantic_map_key": "semantic_map/semantic_map.pb",
        "dataset_meta_key": "meta.json",
        "filter_agents_threshold": 0.5,
    },
    "train_data_loader": {
        "key": "scenes/train.zarr",
        "batch_size": 12,
        "shuffle": True,
        "num_workers": 4,
    },
}


class AgentDataset3d(AgentDataset):
    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        image = sample["image"]
        t_size = (image.shape[0] - 3) // 2 - 1
        # (channels)x(time)x(height)x(width)
        video = np.zeros(
            (5, t_size + 1, image.shape[1], image.shape[2]), dtype=image.dtype
        )
        for i in range(t_size + 1):
            video[0, i, :, :] = image[-3, :, :]
            video[1, i, :, :] = image[-2, :, :]
            video[2, i, :, :] = image[-1, :, :]
            video[3, i, :, :] = image[i, :, :]
            video[4, i, :, :] = image[i + t_size + 1, :, :]
        sample["image"] = video
        return sample


dm = LocalDataManager(None)


def get_loaders(train_batch_size=32, valid_batch_size=64):
    """Prepare loaders.

    Args:
        train_batch_size (int, optional): batch size for training dataset.
            Default is `32`.
        valid_batch_size (int, optional): batch size for validation dataset.
            Default is `64`.

    Returns:
        train and validation data loaders
    """
    rasterizer = build_rasterizer(cfg, dm)

    train_zarr = ChunkedDataset(dm.require("scenes/train.zarr")).open()
    train_dataset = AgentDataset3d(cfg, train_zarr, rasterizer)
    n_samples = len(train_dataset) // 5
    # n_samples = 100
    train_dataset = Subset(train_dataset, list(range(n_samples)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=NUM_WORKERS,
        shuffle=True,
        worker_init_fn=seed_all,
        drop_last=True,
    )
    print(f" * Number of elements in train dataset - {len(train_dataset)}")
    print(f" * Number of elements in train loader - {len(train_loader)}")

    return train_loader, None

    # eval_zarr_path = dm.require("scenes/validate_chopped_100/validate.zarr")
    # eval_gt_path = "scenes/validate_chopped_100/gt.csv"
    # eval_mask_path = "./data/scenes/validate_chopped_100/mask.npz"
    # eval_mask = np.load(eval_mask_path)["arr_0"]

    # valid_zarr = ChunkedDataset(eval_zarr_path).open()
    # valid_dataset = AgentDataset(cfg, valid_zarr, rasterizer)
    # # valid_dataset = Subset(valid_dataset, list(range(200_000)))
    # valid_loader = DataLoader(
    #     valid_dataset,
    #     batch_size=valid_batch_size,
    #     shuffle=False,
    #     num_workers=NUM_WORKERS,
    # )
    # print(f" * Number of elements in valid dataset - {len(valid_dataset)}")
    # print(f" * Number of elements in valid loader - {len(valid_loader)}")

    # return train_loader, valid_loader


def train_fn(
    model,
    loader,
    device,
    loss_fn,
    optimizer,
    scheduler=None,
    accumulation_steps=1,
    verbose=True,
):
    """Train step.
    Args:
        model (nn.Module): model to train
        loader (DataLoader): loader with data
        device (str or torch.device): device to use for placing batches
        loss_fn (nn.Module): loss function, should be callable
        optimizer (torch.optim.Optimizer): model parameters optimizer
        scheduler ([type], optional): batch scheduler to use.
            Default is `None`.
        accumulation_steps (int, optional): number of steps to accumulate gradients.
            Default is `1`.
        verbose (bool, optional): verbosity mode.
            Default is True.
    Returns:
        dict with metics computed during the training on loader
    """
    model.train()
    metrics = {"loss": 0.0}
    with tqdm(total=len(loader), desc="train", disable=not verbose) as progress:
        for idx, batch in enumerate(loader):
            images, targets, target_availabilities = t2d(
                (
                    batch["image"],
                    batch["target_positions"],
                    batch["target_availabilities"],
                ),
                device,
            )

            zero_grad(optimizer)

            predictions, confidences = model(images)
            loss = loss_fn(targets, predictions, confidences, target_availabilities)

            _loss = loss.detach().item()
            metrics["loss"] += _loss

            loss.backward()

            progress.set_postfix_str(f"loss - {_loss:.5f}")
            progress.update(1)

            if (idx + 1) % accumulation_steps == 0:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            if idx == DEBUG:
                break

    metrics["loss"] /= idx + 1
    return metrics


# def valid_fn(model, loader, device, loss_fn, verbose=True):
#     """Validation step.

#     Args:
#         model (nn.Module): model to train
#         loader (DataLoader): loader with data
#         device (str or torch.device): device to use for placing batches
#         loss_fn (nn.Module): loss function, should be callable
#         verbose (bool, optional): verbosity mode.
#             Default is True.

#     Returns:
#         dict with metics computed during the validation on loader
#     """
#     model.eval()
#     metrics = {"loss": 0.0}
#     with torch.no_grad(), tqdm(
#         total=len(loader), desc="valid", disable=not verbose
#     ) as progress:
#         for idx, batch in enumerate(loader):
#             images, targets, target_availabilities = t2d(
#                 (
#                     batch["image"],
#                     batch["target_positions"],
#                     batch["target_availabilities"],
#                 ),
#                 device,
#             )

#             predictions, confidences = model(images)
#             loss = loss_fn(targets, predictions, confidences, target_availabilities)

#             _loss = loss.detach().item()
#             metrics["loss"] += _loss

#             progress.set_postfix_str(f"loss - {_loss:.5f}")
#             progress.update(1)

#             if idx == DEBUG:
#                 break

#     metrics["loss"] /= idx + 1
#     return metrics


def log_metrics(
    stage: str, metrics: dict, logger: TensorboardLogger, loader: str, epoch: int
) -> None:
    """Write metrics to tensorboard and stdout.

    Args:
        stage (str): stage name
        metrics (dict): metrics computed during training/validation steps
        logger (TensorboardLogger): logger to use for storing metrics
        loader (str): loader name
        epoch (int): epoch number
    """
    order = ("loss",)
    for metric_name in order:
        if metric_name in metrics:
            value = metrics[metric_name]
            logger.metric(f"{stage}/{metric_name}", {loader: value}, epoch)
            print(f"{metric_name:>10}: {value:.4f}")


def experiment(logdir, device) -> None:
    """Experiment function

    Args:
        logdir (Path): directory where should be placed logs
        device (str): device name to use
    """
    tb_dir = logdir / "tensorboard"
    main_metric = "loss"
    minimize_metric = True

    seed_all()

    history_n_frames = cfg["model_params"]["history_num_frames"]
    future_n_frames = cfg["model_params"]["future_num_frames"]
    n_trajectories = 3
    model = ModelWithConfidence(
        backbone=resnet18_3d(
            in_channels=3 + 2,  # 3 + 2 * (history_n_frames + 1),
            num_classes=2 * future_n_frames * n_trajectories + n_trajectories,
        ),
        future_num_frames=future_n_frames,
        num_trajectories=n_trajectories,
    )
    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = neg_multi_log_likelihood_batch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    with TensorboardLogger(tb_dir) as tb:
        stage = "stage_0"
        n_epochs = 1
        print(f"Stage - {stage}")

        checkpointer = CheckpointManager(
            logdir=logdir / stage,
            metric=main_metric,
            metric_minimization=minimize_metric,
            save_n_best=5,
        )

        train_loader, valid_loader = get_loaders(
            train_batch_size=32, valid_batch_size=32
        )

        for epoch in range(1, n_epochs + 1):
            epoch_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{epoch_start_time}]\n[Epoch {epoch}/{n_epochs}]")

            train_metrics = train_fn(model, train_loader, device, criterion, optimizer)
            log_metrics(stage, train_metrics, tb, "train", epoch)

            valid_metrics = train_metrics
            # valid_metrics = valid_fn(model, valid_loader, device, criterion)
            # log_metrics(stage, valid_metrics, tb, "valid", epoch)

            checkpointer.process(
                metric_value=valid_metrics[main_metric],
                epoch=epoch,
                checkpoint=make_checkpoint(
                    stage,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    metrics={"train": train_metrics, "valid": valid_metrics},
                ),
            )

            scheduler.step()


def main() -> None:
    experiment_name = "test_resnet18_3d_confidence"
    logdir = Path(".") / "logs" / experiment_name

    if not torch.cuda.is_available():
        raise ValueError("Something went wrong - CUDA devices is not available!")

    device = torch.device("cuda:0")

    if logdir.is_dir():
        shutil.rmtree(logdir, ignore_errors=True)
        print(f" * Removed existing directory with logs - '{logdir}'")

    experiment(logdir, device)


if __name__ == "__main__":
    main()
