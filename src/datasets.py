import os
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDatase
from l5kit.rasterization import build_rasterizer
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
