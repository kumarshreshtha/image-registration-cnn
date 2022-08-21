"""dataset module for dl image registration training."""

import pathlib

import numpy as np
import torch
import torchio
from torch.utils import data


class RegDataset(data.Dataset):
    """Minimal dataset class that returns the source and target volume pairs.

    The datadir is expected to contain input volumes as numpy arrays on disk:
     - ./{uid}_src.npy
     - ./{uid}_tgt.npy
    """

    def __init__(self, data_dir: pathlib.Path, target_shape=(64, 192, 192)):
        super().__init__()
        self._data_dir = data_dir
        self.uids = {fname.stem.split("_src")
                     for fname in self._data_dir.glob("*_src.npy")}
        self.transforms = torchio.Compose([torchio.Resize(target_shape),
                                           torchio.Clamp(0, 1300),
                                           torchio.RescaleIntensity((0, 1))])

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        src = torch.from_numpy(
            np.load(self._data_dir/f"{self.uids[idx]}_src.npy"))
        tgt = torch.from_numpy(
            np.load(self._data_dir/f"{self.uids[idx]}_tgt.npy"))
        return self.transforms(src), self.transforms(tgt)