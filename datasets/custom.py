from pathlib import Path
import re

import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ShapeDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.dir_path = Path(dir_path)
        assert self.dir_path.exists(), f"Directory {dir_path} does not exist."
        assert self.dir_path.is_dir(), f"{dir_path} is not a directory."
        self.data_paths = sorted(list(self.dir_path.glob('*.png')))
        assert len(self.data_paths) > 0, f"No data found in {dir_path}."

        self.data = np.stack([self.load_data(path) for path in self.data_paths], axis=0)
        self.labels = np.array([self.load_label(path) for path in self.data_paths], dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def load_data(self, path):
        # Load in [0, 255] range
        data = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        # data = data.astype(np.float32)
        data = np.expand_dims(data, axis=-1)
        return data

    def load_label(self, path):
        label = re.search(r'poly(\d+)_.*png', str(path)).group(1)
        label = int(label)
        return label

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            data = self.transform(data)
        return data, label
