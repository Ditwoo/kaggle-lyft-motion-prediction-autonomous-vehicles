import os
from pathlib import Path
import numpy as np

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.evaluation import create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.rasterization import build_rasterizer


os.environ["L5KIT_DATA_FOLDER"] = "./data"

cfg = {
    "format_version": 4,
    "model_params": {
        "history_num_frames": 10,
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


def main():
    dm = LocalDataManager(None)
    rasterizer = build_rasterizer(cfg, dm)

    num_frames_to_chop = 100
    eval_base_path = create_chopped_dataset(
        dm.require("scenes/validate.zarr"),
        cfg["raster_params"]["filter_agents_threshold"],
        num_frames_to_chop,
        cfg["model_params"]["future_num_frames"],
        MIN_FUTURE_STEPS,
    )

    print("Path:", eval_base_path)


if __name__ == "__main__":
    main()
