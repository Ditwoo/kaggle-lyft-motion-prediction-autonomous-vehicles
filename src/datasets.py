import os
from datetime import datetime
import torch
import numpy as np
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from torch.utils.data import Dataset
from l5kit.geometry import transform_points
from l5kit.visualization.utils import draw_trajectory


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

def acceleration_approx(smaple: np.ndarray, h: float = 1.0, mean: float = 0.00279, std: float = 1.40201):
    x = smaple[:, 0]
    y = smaple[:, 1]
    x_acc = (((x[:-2] - 2 * x[1:-1] + x[2:]) / h ** 2) - mean) / std
    y_acc = (((y[:-2] - 2 * y[1:-1] + y[2:]) / h ** 2) - mean) / std
    
    return np.concatenate([x_acc, y_acc])

class AccelAgentDataset(AgentDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample['xy_acceleration'] = acceleration_approx(sample['history_positions'])
        return sample