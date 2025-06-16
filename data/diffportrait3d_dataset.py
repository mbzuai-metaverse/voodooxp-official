import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import json
import numpy as np
import os.path as osp
import random
from zipfile import ZipFile
# try:
#     from turbojpeg import TurboJPEG
# except ImportError:
print('Warning: PNG format. Will use cv2 to read image instead')
TurboJPEG = None

from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class DiffPortrait3dDataset(Dataset):
    def __init__(
        self,
        data_root,
        image_raw_size=128
    ):
        super().__init__()

        self.image_raw_size = image_raw_size
        self._data_root = data_root

        if (not osp.isfile(self._data_root)) and (not osp.isfile(self._data_root)):
            raise ValueError(f'Data root is not a valid folder or file')

        if osp.isdir(self._data_root):
            self._datatype = 'dir'
        elif self._data_root.lower().endswith('.zip'):
            self._datatype = 'zip'
            if TurboJPEG is not None:
                self.jpeg_decoder = TurboJPEG()
            else:
                self.jpeg_decoder = None
        else:
            raise ValueError(f'Data type is not supported')

        # Transformation including to_tensor and augmentations.
        #
        # Avoid using geometric augmentations here because it can break
        # The alignment of the images.
        #
        # Image is normalized to [-1, 1]
        self.transform = A.Compose(
            [
                A.ColorJitter(),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.),
                A.Resize(512, 512),
                ToTensorV2(),
            ],
            additional_targets={'xd_image': 'image'}
        )
        self.downsample = A.Resize(self.image_raw_size, self.image_raw_size)

        # Grouping images
        self.frame_groups_recon = {}
        self.frame_groups_reenact = {}

        with ZipFile(self._data_root).open('labels.json') as f:
            labels = json.load(f)
        
        labels = labels['labels']
        for k, v in labels.items():
            k = k.strip('/')
            
            group_name_recon = self._get_group_from_path(k)

            if group_name_recon not in self.frame_groups_recon:
                self.frame_groups_recon[group_name_recon] = []
            self.frame_groups_recon[group_name_recon].append({
                'image_path': k,
                'label': v
            })

        # Only keep groups with 2 or more elements
        self.group_names = []
        for k, v in self.frame_groups_recon.items():
            if len(v) > 1:
                self.group_names.append(k)

        # Zip data placeholder
        self._zipfile = None

    @staticmethod
    def _get_group_from_path(path):
        identity_idx, category_idx, cam_idx, frame_idx = path.split('/')
        return '_'.join((identity_idx, category_idx, cam_idx))

    def _get_zipfile(self):
        if self._zipfile is None:
            self._zipfile = ZipFile(self._data_root)
        return self._zipfile

    def _read_image(self, image_path):
        if self._datatype == 'dir':
            image = cv2.cvtColor(
                cv2.imread(osp.join(self._data_root, image_path)),
                cv2.COLOR_BGR2RGB
            )
        elif self._datatype == 'zip':
            with self._get_zipfile().open(image_path, 'r') as f:
                if self.jpeg_decoder is not None:
                    image = self.jpeg_decoder.decode(f.read())
                else:
                    image_array = np.frombuffer(f.read(), dtype=np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            assert 0

        return image

    def _extract_from_labels(self, all_labels):
        images = {}
        for k, v in all_labels.items():
            images[k] = self._read_image(v['image_path'])
        
        images = self.transform(**images)

        out = []
        for k, v in all_labels.items():
            data = {
                'image': images[k],
                'image_raw': F.interpolate(
                    images[k].unsqueeze(0), (self.image_raw_size, self.image_raw_size)
                ).squeeze(0),
                'cam2world': np.array(v['label'][:16]).reshape(4, 4).astype(np.float32),
                'intrinsics': np.array(v['label'][16:]).reshape(3, 3).astype(np.float32),
            }
            out.append(data)

        return out

    def __getitem__(self, group_idx):
        group_name = self.group_names[group_idx]
        
        xs_data, xd_data = random.sample(self.frame_groups_recon[group_name], 2)
        xs, xd = self._extract_from_labels({'image': xs_data, 'xd_image': xd_data})

        out = {
            'xs_data': xs,
            'xd_data': xd,
        }

        original_image_path = osp.join(osp.dirname(xd_data['image_path']), 'origin.png')
        original_image = self._read_image(original_image_path)
        original_image = self.transform(image=original_image)['image']
        out['original_image'] = F.interpolate(
            original_image.unsqueeze(0), (self.image_raw_size, self.image_raw_size)
        ).squeeze(0)

        out['triplane_weight'] = torch.ones([1])

        return out

    def __len__(self):
        return len(self.group_names)

