import os
import time
import shutil
from pathlib import Path
import numpy as np
from functools import partial


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader, RandomSampler
import torch.nn.functional as F

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.evaluation import create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.geometry import transform_points
from l5kit.evaluation import write_pred_csv, compute_metrics_csv
from l5kit.evaluation.metrics import neg_multi_log_likelihood

from src.experiments.baseline_resnet_fast_rasterizer.l5kit_modified.rasterizer.build_rasterizer import (
    build_rasterizer,
)

from src.batteries import (
    seed_all,
    t2d,
    zero_grad,
    CheckpointManager,
    TensorboardLogger,
    make_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from src.batteries.progress import tqdm
from src.models import ModelWithConfidence
from src.models.resnets import resnet34_accel
from src.criterion import neg_multi_log_likelihood_batch
from src.datasets import AccelAgentDataset

torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


DEBUG = int(os.environ.get("DEBUG", -1))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "16"))
os.environ["L5KIT_DATA_FOLDER"] = "./data"

cfg = {
    "format_version": 4,
    "model_params": {
        "history_num_frames": 25,
        "history_step_size": 1,
        "history_delta_time": 0.1,
        "future_num_frames": 50,
        "future_step_size": 1,
        "future_delta_time": 0.1,
    },
    "raster_params": {
        "raster_size": [512, 512],
        "pixel_size": [0.25, 0.25],
        "ego_center": [0.25, 0.5],
        "map_type": "py_semantic",
        "satellite_map_key": "aerial_map/aerial_map.png",
        "semantic_map_key": "semantic_map/semantic_map.pb",
        "dataset_meta_key": "meta.json",
        "filter_agents_threshold": 0.5,
    },
    "train_data_loader": {
        "key": "scenes/train.zarr",
        "batch_size": 32,
        "shuffle": True,
        "num_workers": 4,
    },
}


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
    DATASET_CLASS = AccelAgentDataset

    train_zarr = ChunkedDataset(dm.require("scenes/train.zarr")).open()
    train_dataset = DATASET_CLASS(cfg, train_zarr, rasterizer)
    # indices = np.arange(22079968, 22496709, 1)
    # train_dataset = Subset(train_dataset, indices)

    # sizes = ps.read_csv(os.environ["TRAIN_TRAJ_SIZES"])["size"].values
    # is_small = sizes < 6
    # n_points = is_small.sum()
    # to_sample = n_points // 4
    # print(" * points - {} (points to sample - {})".format(n_points, to_sample))
    # print(" * paths  -", sizes.shape[0] - n_points)
    # indices = np.concatenate(
    #     [
    #         np.random.choice(np.where(is_small)[0], size=to_sample, replace=False,),
    #         np.where(~is_small)[0],
    #     ]
    # )
    # train_dataset = Subset(train_dataset, indices)

    # n_samples = len(train_dataset) // 2
    # train_dataset = Subset(train_dataset, list(range(n_samples)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=NUM_WORKERS,
        shuffle=True,
        worker_init_fn=seed_all,
        drop_last=True,
    )
    # train_loader = BatchPrefetchLoaderWrapper(train_loader, num_prefetches=6)
    print(f" * Number of elements in train dataset - {len(train_dataset)}")
    print(f" * Number of elements in train loader - {len(train_loader)}")

    valid_zarr_path = dm.require("scenes/validate_chopped_100/validate.zarr")
    mask_path = dm.require("scenes/validate_chopped_100/mask.npz")
    valid_mask = np.load(mask_path)["arr_0"]
    valid_gt_path = dm.require("scenes/validate_chopped_100/gt.csv")

    valid_zarr = ChunkedDataset(valid_zarr_path).open()
    valid_dataset = DATASET_CLASS(cfg, valid_zarr, rasterizer, agents_mask=valid_mask)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    print(f" * Number of elements in valid dataset - {len(valid_dataset)}")
    print(f" * Number of elements in valid loader - {len(valid_loader)}")

    return train_loader, (valid_loader, valid_gt_path)


def train_fn(
    model,
    loader,
    device,
    loss_fn,
    optimizer,
    scheduler=None,
    accumulation_steps=1,
    verbose=True,
    tensorboard_logger=None,
    logdir=None,
    validation_fn=None,
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
    n_batches = len(loader)

    indices_to_save = [int(n_batches * pcnt) for pcnt in np.arange(0.1, 1, 0.1)]
    last_score = 0.0

    with tqdm(total=len(loader), desc="train", disable=not verbose) as progress:
        for idx, batch in enumerate(loader):
            (images, targets, target_availabilities, acceleration,) = t2d(
                (
                    batch["image"],
                    batch["target_positions"],
                    batch["target_availabilities"],
                    batch["xy_acceleration"],
                ),
                device,
            )

            zero_grad(optimizer)

            predictions, confidences = model(images, acceleration)
            loss = loss_fn(targets, predictions, confidences, target_availabilities)

            _loss = loss.detach().item()

            metrics["loss"] += _loss

            if (idx + 1) % 30_000 == 0 and validation_fn is not None:
                score = validation_fn(model=model, device=device)
                model.train()
                last_score = score

                if logdir is not None:
                    checkpoint = make_checkpoint("train", idx + 1, model)
                    save_checkpoint(checkpoint, logdir, f"train_{idx}.pth")
            else:
                score = None

            if tensorboard_logger is not None:

                tensorboard_logger.metric("loss", _loss, idx)

            loss.backward()

            progress.set_postfix_str(
                f"loss - {_loss:.5f}"
                # f"loss - {_loss:.5f}"
                f"loss - {_loss:.5f}, score - {last_score:.5f}"
            )
            progress.update(1)

            if (idx + 1) % accumulation_steps == 0:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            if idx == DEBUG:
                break

    for k in metrics.keys():
        metrics[k] /= idx + 1
    return metrics


def valid_fn(model, loader, device, ground_truth_file, logdir, verbose=True):
    """Validation step.

    Args:
        model (nn.Module): model to train
        loader (DataLoader): loader with data
        device (str or torch.device): device to use for placing batches
        loss_fn (nn.Module): loss function, should be callable
        verbose (bool, optional): verbosity mode.
            Default is True.

    Returns:
        dict with metics computed during the validation on loader
    """
    model.eval()

    future_coords_offsets_pd = []
    timestamps = []
    confidences_list = []
    agent_ids = []

    with torch.no_grad(), tqdm(
        total=len(loader), desc="valid", disable=not verbose
    ) as progress:
        for idx, batch in enumerate(loader):
            images, acceleration = t2d(
                [batch["image"], batch["xy_acceleration"]], device
            )

            predictions, confidences = model(images, acceleration)

            _gt = batch["target_positions"].cpu().numpy().copy()
            predictions = predictions.cpu().numpy().copy()
            world_from_agents = batch["world_from_agent"].numpy()
            centroids = batch["centroid"].numpy()

            for idx in range(len(predictions)):
                for mode in range(3):
                    # FIX
                    predictions[idx, mode, :, :] = (
                        transform_points(
                            predictions[idx, mode, :, :], world_from_agents[idx]
                        )
                        - centroids[idx][:2]
                    )
                _gt[idx, :, :] = (
                    transform_points(_gt[idx, :, :], world_from_agents[idx])
                    - centroids[idx][:2]
                )

            future_coords_offsets_pd.append(predictions.copy())
            confidences_list.append(confidences.cpu().numpy().copy())
            timestamps.append(batch["timestamp"].numpy().copy())
            agent_ids.append(batch["track_id"].numpy().copy())

            progress.update(1)

            if idx == DEBUG:
                break

    predictions_file = str(logdir / "preds_validate_chopped.csv")
    write_pred_csv(
        predictions_file,
        timestamps=np.concatenate(timestamps),
        track_ids=np.concatenate(agent_ids),
        coords=np.concatenate(future_coords_offsets_pd),
        confs=np.concatenate(confidences_list),
    )

    metrics = compute_metrics_csv(
        ground_truth_file,
        predictions_file,
        [neg_multi_log_likelihood],
    )

    return {"score": metrics["neg_multi_log_likelihood"]}


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
    order = ("loss", "score", "mask_loss", "regression_loss")
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
        backbone=resnet34_accel(
            pretrained=True,
            in_channels=3 + 3,
            num_classes=2 * future_n_frames * n_trajectories + n_trajectories,
            in_accel_features=(history_n_frames - 1) * 2,
            num_accel_features=32,
        ),
        future_num_frames=future_n_frames,
        num_trajectories=n_trajectories,
    )

    load_checkpoint(
        "./logs/resnet34_frast_fulldata_confidence_25hist_accel/epoch_1/train_689999.pth",
        model,
    )

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

        train_loader, (valid_loader, valid_gt_path) = get_loaders(
            train_batch_size=32, valid_batch_size=32
        )

        valid_func = partial(
            valid_fn,
            loader=valid_loader,
            ground_truth_file=valid_gt_path,
            logdir=logdir,
            verbose=True,
        )

        for epoch in range(1, n_epochs + 1):
            epoch_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{epoch_start_time}]\n[Epoch {epoch}/{n_epochs}]")

            # try:
            train_metrics = train_fn(
                model,
                train_loader,
                device,
                criterion,
                optimizer,
                tensorboard_logger=tb,
                logdir=logdir / f"epoch_{epoch}",
                validation_fn=valid_func,
            )
            log_metrics(stage, train_metrics, tb, "train", epoch)
            # except BaseException:
            # train_metrics = {"message": "An exception occured!"}

            valid_metrics = valid_fn(model, valid_loader, device, valid_gt_path, logdir)
            log_metrics(stage, valid_metrics, tb, "valid", epoch)

            checkpointer.process(
                metric_value=valid_metrics["score"],
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

        # TODO: Закинуть submit fn и проверить на сабсете


def main() -> None:
    experiment_name = "resnet34_frast_fulldata_confidence_25hist_accel_cnt"
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
