import torch.nn.functional as F
from torch.utils.data import Dataset
from data import get_dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import numpy as np
import json
import os.path as osp
from zipfile import ZipFile
import random

from utils.registry import DATASET_REGISTRY


GENERIC_FACE_PATH = 'datasets/nersemble_generic_face.zip'


class GenericFaceSampler:
    def __init__(self):
        # Zip data placeholder
        self._zipfile = None

        with ZipFile(GENERIC_FACE_PATH).open('labels.json') as f:
            labels = json.load(f)

        all_groups = {}
        for k, v in labels.items():
            group_name = self._get_group_from_path(k)
            if group_name not in all_groups:
                all_groups[group_name] = []
            all_groups[group_name].append((k, v))

        self.all_groups = list(all_groups.values())
            
        self.transform = A.Compose(
            [
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.),
                ToTensorV2(),
            ],
            additional_targets={'xd_image': 'image'}
        )

    @staticmethod
    def _get_group_from_path(path):
        identity_idx, category_idx, cam_idx, frame_idx = path.split('/')
        return '_'.join((identity_idx, category_idx, frame_idx))

    def _get_zipfile(self):
        if self._zipfile is None:
            self._zipfile = ZipFile(GENERIC_FACE_PATH)
        return self._zipfile

    def _read_image(self, path):
        with self._get_zipfile().open(path, 'r') as f:
            image_array = np.frombuffer(f.read(), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def __call__(self):
        group_idx = np.random.randint(len(self.all_groups))
        metadata1, metadata2 = random.sample(self.all_groups[group_idx], 2)
        image1 = self._read_image(metadata1[0])
        image2 = self._read_image(metadata2[0])

        image1 = self.transform(image=image1)['image']
        image2 = self.transform(image=image2)['image']
        image1_raw = F.interpolate(image1.unsqueeze(0), (128, 128)).squeeze(0)

        return {
            'image': image1,
            'image_raw': image1_raw,
            'exp_image': image2,
            'cam2world': np.array(metadata1[1][:16]).reshape(4, 4).astype(np.float32),
            'intrinsics': np.array(metadata1[1][16:]).reshape(3, 3).astype(np.float32)
        }


@DATASET_REGISTRY.register()
class ConcatDataset(Dataset):
    def __init__(self, use_generic_face, **all_data_opts):
        super().__init__()

        self.use_generic_face = use_generic_face
        self.all_datasets = []
        self.all_sampling_weights = []
        self.num_samples = 0
        for data_opt in all_data_opts.values():
            print(f"Loading {data_opt['data_class']}")
            dataset = get_dataset(data_opt)
            self.all_datasets.append(dataset)
            self.all_sampling_weights.append(data_opt['sampling_weight'])
            self.num_samples += len(dataset)

        assert sum(self.all_sampling_weights) == 1, 'Sum of ratios must be equal to 1!'

        if self.use_generic_face:
            self.generic_face_sampler = GenericFaceSampler()

        self.previous_data = None

    def __getitem__(self, _):
        dataset_idx = np.random.choice(range(len(self.all_sampling_weights)), p=self.all_sampling_weights)
        dataset = self.all_datasets[dataset_idx]

        sample_idx = np.random.randint(len(dataset))

        out = dataset[sample_idx]

        if self.previous_data is not None and np.random.randint(2) == 0:
            out['xs_aux_data'] = self.previous_data['xs_aux_data']
            out['xd_aux_data'] = self.previous_data['xd_aux_data']

        self.previous_data = out

        if self.use_generic_face:
            out['generic_data'] = self.generic_face_sampler()

        return out

    def __len__(self):
        return self.num_samples
