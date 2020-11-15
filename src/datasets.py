import os
from datetime import datetime

import cv2
import numpy as np
import torch
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.geometry import transform_points
from l5kit.rasterization import build_rasterizer
from l5kit.visualization.utils import draw_trajectory
from torch.utils.data import Dataset


class MotionDataset(Dataset):
    def __init__(
        self, cfg, loader_key="train_data_loader", fn_rasterizer=build_rasterizer,
    ):
        self.cfg = cfg
        self.loader_key = loader_key
        self.fn_rasterizer = fn_rasterizer

        self.setup()

    def setup(self):
        self.dm = LocalDataManager(None)
        self.rasterizer = self.fn_rasterizer(self.cfg, self.dm)
        self.data_zarr = ChunkedDataset(
            self.dm.require(self.cfg[self.loader_key]["key"])
        ).open()

        self.ds = AgentDataset(self.cfg, self.data_zarr, self.rasterizer)

    def __getitem__(self, index):
        return self.ds[index]

    def __len__(self):
        return len(self.ds)


i_global_to_local = {
    16: 1,
    17: 2,
    38: 3,
    39: 4,
    61: 5,
    83: 6,
    84: 7,
    106: 8,
    129: 9,
    151: 10,
    152: 11,
    174: 12,
    175: 13,
    196: 14,
    197: 15,
    218: 16,
    219: 17,
    237: 18,
    238: 19,
    239: 20,
    258: 21,
    259: 22,
    260: 23,
    278: 24,
    279: 25,
    280: 26,
    299: 27,
    300: 28,
    301: 29,
    322: 30,
    323: 31,
    344: 32,
    345: 33,
    367: 34,
    368: 35,
    390: 36,
    391: 37,
    412: 38,
    413: 39,
    414: 40,
    435: 41,
    436: 42,
    457: 43,
    458: 44,
    478: 45,
    479: 46,
    499: 47,
    500: 48,
    520: 49,
    521: 50,
    541: 51,
    542: 52,
    543: 53,
    562: 54,
    563: 55,
    564: 56,
    583: 57,
    584: 58,
    585: 59,
    605: 60,
    606: 61,
    626: 62,
    627: 63,
    647: 64,
    648: 65,
    668: 66,
    669: 67,
    689: 68,
    690: 69,
    691: 70,
    710: 71,
    711: 72,
    712: 73,
    731: 74,
    732: 75,
    733: 76,
    752: 77,
    753: 78,
    754: 79,
    773: 80,
    774: 81,
    775: 82,
    794: 83,
    795: 84,
    796: 85,
    816: 86,
    817: 87,
    836: 88,
    837: 89,
    838: 90,
    858: 91,
    859: 92,
    860: 93,
    881: 94,
    882: 95,
}


def to_flatten_square_idx(sample):
    # creating ids mapping:
    # from pprint import pprint
    # dataset = train
    # ids = [to_flatten_square_idx(dataset[i]) for i in tqdm(np.random.permutation(len(dataset))[:10_000])]
    # sorted_ids = sorted(list(set(ids)))
    # pprint(dict(zip(sorted_ids, range(1, len(sorted_ids) + 1))))
    # plt.figure(figsize=(12,12))
    # plt.xlim(-11, 11)
    # plt.ylim(-26, 16)
    # for i in np.random.permutation(len(test))[:50]:
    #     x, y = test[i]['centroid']
    #     ids = to_flatten_square_idx(test[i])
    #     plt.text(x // 100, y // 100, ids)
    # plt.show()
    # cell size is 100
    x_bounds = (-11, 11)  # / 100
    y_bounds = (-26, 16)  # / 100
    n_cols = x_bounds[1] - x_bounds[0]

    x0, y0 = sample["centroid"]
    x0, y0 = x0 // 100, y0 // 100

    i_col = x0 - x_bounds[0]
    i_row = y0 - y_bounds[0]

    i_global = int(n_cols * max(i_row - 1, 0) + i_col)
    return i_global_to_local.get(i_global, 0)


def parse_timestamp(timestamp):
    d = datetime.fromtimestamp(timestamp / 10e9)
    return d.hour, d.weekday(), d.month - 1


class CubicAgentDataset(AgentDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)

        sample["square_category"] = torch.tensor(to_flatten_square_idx(sample)).long()

        hour, weekday, month = parse_timestamp(sample["timestamp"])
        sample["time_hour"] = torch.tensor(hour).long()
        sample["time_weekday"] = torch.tensor(weekday).long()
        sample["time_month"] = torch.tensor(month).long()
        return sample


class SegmentationAgentDataset(AgentDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        mask = np.zeros(sample["image"].shape[1:], dtype=np.uint8)
        points = transform_points(
            sample["target_positions"], sample["raster_from_agent"]
        )
        points = points[sample["target_availabilities"].astype(bool)]
        draw_trajectory(mask, points, (255))
        mask = torch.from_numpy((mask / 255.0).astype(np.float32)).unsqueeze(0)
        sample["mask"] = mask
        sample["square_category"] = torch.tensor(to_flatten_square_idx(sample)).long()
        return sample


def get_num_channels(image):
    return image.shape[2] if len(image.shape) == 3 else 1


def _maybe_process_in_chunks(process_fn, **kwargs):
    """
    Wrap OpenCV function to enable processing images with more than 4 channels.
    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.

    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.

    Returns:
        numpy.ndarray: Transformed image.
    """

    def __process_fn(img):
        num_channels = get_num_channels(img)
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == 2:
                    # Many OpenCV functions cannot work with 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i : index + i + 1]
                        chunk = process_fn(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index : index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn


def rotate(
    img,
    angle,
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_REFLECT_101,
    value=None,
):
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)

    warp_fn = _maybe_process_in_chunks(
        cv2.warpAffine,
        M=matrix,
        dsize=(width, height),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )
    return warp_fn(img)


class RotationAgentDataset(AgentDataset):
    rotation_angles = (-15, 15)
    rotation_probability = 0.3

    def __getitem__(self, index):
        sample = super().__getitem__(index)

        if np.random.uniform() <= self.rotation_probability:
            angle = np.random.randint(*self.rotation_angles)
            theta = np.radians(angle)
            c = np.cos(theta)
            s = np.sin(theta)
            rot_matrix = np.array([[c, s], [-s, c]])

            image = np.moveaxis(sample["image"], 0, -1)  # CxHxW -> HxWxC
            sample["image"] = np.moveaxis(rotate(image, angle), -1, 0)  # HxWxC -> CxHxW
            sample["target_positions"] = sample["target_positions"].dot(rot_matrix)

        return sample
