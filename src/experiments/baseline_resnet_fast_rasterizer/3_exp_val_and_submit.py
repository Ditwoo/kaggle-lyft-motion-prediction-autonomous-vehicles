import os
import sys
import time
import shutil
from pathlib import Path
import numpy as np


import bisect
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

import l5kit
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
from pathlib import Path

from src.batteries import t2d, load_checkpoint
from src.batteries.progress import tqdm
from src.models.genet import genet_normal
from src.models.resnets import resnet18
from src.models import ModelWithConfidence
from src.criterion import neg_multi_log_likelihood_batch

from src.experiments.baseline_resnet_fast_rasterizer.l5kit_modified.rasterizer.build_rasterizer import build_rasterizer

os.environ["L5KIT_DATA_FOLDER"] = "./data"
DATA_DIR = './data'

DEBUG = int(os.environ.get("DEBUG", -1))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "16"))

exp_basename = "resnet18_fast_rasterizer_25data_confidence_25hist"
checkpoint_path = f"./logs/{exp_basename}/stage_0/best.pth"
val_predictions_file = f"val_predictions/preds_validate_chopped_100_{exp_basename}.csv"
submission_file = f"submissions/{exp_basename}.csv"

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
        "batch_size": 32,
        "shuffle": True,
        "num_workers": 4,
    },
}


future_n_frames = cfg["model_params"]["future_num_frames"]
n_trajectories = 3
model = ModelWithConfidence(
    backbone=resnet18(
        pretrained=True,
        in_channels=6,
        num_classes=2 * future_n_frames * n_trajectories + n_trajectories,
    ),
    future_num_frames=future_n_frames,
    num_trajectories=n_trajectories,
)

load_checkpoint(checkpoint_path, model)
model = model.eval()

device = torch.device("cuda:0")
model = model.to(device)

valid_mask = np.load(f"{DATA_DIR}/scenes/validate_chopped_100/mask.npz")["arr_0"]

dm = LocalDataManager(DATA_DIR)


rasterizer = build_rasterizer(cfg, dm)

valid_zarr = ChunkedDataset(dm.require("scenes/validate_chopped_100/validate.zarr")).open()

bs = 32

valid_dataset = AgentDataset(cfg, valid_zarr, rasterizer, agents_mask=valid_mask)
print(len(valid_dataset))

# valid_dataset = Subset(valid_dataset, list(range(bs * 4)))

valid_dataloader = DataLoader(
    valid_dataset,
    shuffle=False,
    batch_size=bs,
    num_workers=30,
)

model.eval()
torch.set_grad_enabled(False)

# store information for evaluation
future_coords_offsets_pd = []
ground_truth = []
timestamps = []
confidences_list = []
agent_ids = []

with tqdm(total=len(valid_dataloader)) as progress:
    for batch in valid_dataloader:
        inputs = batch['image'].to(device)

        preds, confidences = model(inputs)
        
        # TODO: fix coordinates
        _gt = batch["target_positions"].cpu().numpy().copy()
        preds = preds.cpu().numpy().copy()
        world_from_agents = batch["world_from_agent"].numpy()
        centroids = batch["centroid"].numpy()
        for idx in range(len(preds)):
            for mode in range(n_trajectories):
                # FIX
                preds[idx, mode, :, :] = transform_points(preds[idx, mode, :, :], world_from_agents[idx]) - centroids[idx][:2]
            _gt[idx, :, :] = transform_points(_gt[idx, :, :], world_from_agents[idx]) - centroids[idx][:2]
        
        future_coords_offsets_pd.append(preds.copy())
        confidences_list.append(confidences.cpu().numpy().copy())
        timestamps.append(batch["timestamp"].numpy().copy())
        agent_ids.append(batch["track_id"].numpy().copy())

        progress.update(1)

write_pred_csv(
    val_predictions_file,
    timestamps=np.concatenate(timestamps),
    track_ids=np.concatenate(agent_ids),
    coords=np.concatenate(future_coords_offsets_pd),
    confs=np.concatenate(confidences_list)
)

metrics = compute_metrics_csv(
    f"{DATA_DIR}/scenes/validate_chopped_100/gt.csv",
    val_predictions_file,
    [neg_multi_log_likelihood, time_displace],
)

for metric_name, metric_mean in metrics.items():
    print(metric_name, metric_mean)


#====== INIT TEST DATASET=============================================================
rasterizer = build_rasterizer(cfg, dm)
test_zarr = ChunkedDataset(dm.require("scenes/test.zarr")).open()
test_mask = np.load(f"{DATA_DIR}/scenes/mask.npz")["arr_0"]
test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
test_dataloader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=32,
                             num_workers=30)

model.eval()
torch.set_grad_enabled(False)

# store information for evaluation
future_coords_offsets_pd = []
ground_truth = []
timestamps = []
confidences_list = []
agent_ids = []

with tqdm(total=len(test_dataloader)) as progress:
    for batch in test_dataloader:
        inputs = batch['image'].to(device)

        preds, confidences = model(inputs)
        
        # TODO: fix coordinates
        _gt = batch["target_positions"].cpu().numpy().copy()
        preds = preds.cpu().numpy().copy()
        world_from_agents = batch["world_from_agent"].numpy()
        centroids = batch["centroid"].numpy()
        for idx in range(len(preds)):
            for mode in range(n_trajectories):
                # FIX
                preds[idx, mode, :, :] = transform_points(preds[idx, mode, :, :], world_from_agents[idx]) - centroids[idx][:2]
            _gt[idx, :, :] = transform_points(_gt[idx, :, :], world_from_agents[idx]) - centroids[idx][:2]
        
        future_coords_offsets_pd.append(preds.copy())
        confidences_list.append(confidences.cpu().numpy().copy())
        timestamps.append(batch["timestamp"].numpy().copy())
        agent_ids.append(batch["track_id"].numpy().copy())

        progress.update(1)
        
        
write_pred_csv(
    submission_file,
    timestamps=np.concatenate(timestamps),
    track_ids=np.concatenate(agent_ids),
    coords=np.concatenate(future_coords_offsets_pd),
    confs=np.concatenate(confidences_list)
)  
        
        
        
        
        
