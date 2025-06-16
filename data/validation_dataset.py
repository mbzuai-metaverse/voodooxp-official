from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import json
import numpy as np
import os.path as osp
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ValidationDataset(Dataset):
    def __init__(self, data_root, debug):
        super().__init__()

        self.data_root = data_root

        label_path = osp.join(self.data_root, 'labels.json')
        with open(label_path, 'r') as f:
            self.labels = json.load(f)

        self.source_keys = []
        self.driver_keys = []
        for k in self.labels:
            if 'source' in k:
                self.source_keys.append(k)
            else:
                self.driver_keys.append(k)

        assert len(self.source_keys) == len(self.driver_keys)
        self.source_keys.sort()
        self.driver_keys.sort()

        if debug:
            self.source_keys = self.source_keys[:3]
            self.driver_keys = self.driver_keys[:3]

        self.transform = A.Compose(
            [
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.),
                A.Resize(512, 512),
                ToTensorV2(),
            ],
        )

    def _load_data_from_label(self, key):
        image_path = osp.join(self.data_root, key)
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']

        cam2world = np.array(self.labels[key][:16]).reshape(4, 4).astype(np.float32)
        intrinsics = np.array(self.labels[key][16:]).reshape(3, 3).astype(np.float32)

        return {
            'image': img,
            'intrinsics': intrinsics,
            'cam2world': cam2world
        }

    def __getitem__(self, idx):
        source_key = self.source_keys[idx]
        driver_key = self.driver_keys[idx]

        return {
            'xs_data': self._load_data_from_label(source_key),
            'xd_data': self._load_data_from_label(driver_key),
            'xs_key': source_key,
            'xd_key': driver_key
        }

    def __len__(self):
        return len(self.source_keys)
