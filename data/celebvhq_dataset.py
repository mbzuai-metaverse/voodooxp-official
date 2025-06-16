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
try:
    from turbojpeg import TurboJPEG
except ImportError:
    print('Warning: turbojpeg is not installed. Will use cv2 to read image instead')
    TurboJPEG = None

from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class CelebVHQDataset(Dataset):
    def __init__(
        self,
        data_root,
        mode,
        image_raw_size=128,
    ):
        super().__init__()

        self._mode = mode
        assert self._mode in ['reenactment']

        self._data_root = data_root
        self.image_raw_size = image_raw_size

        if osp.isdir(self._data_root):
            self._datatype = 'dir'
        elif self._data_root.lower().endswith('.zip'):
            self._datatype = 'zip'
            if TurboJPEG is not None:
                self.jpeg_decoder = TurboJPEG()
            else:
                self.jpeg_decoder = None

            # Zip data placeholder
            self._zipfile = None
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
        self.downsample = A.Resize(image_raw_size, image_raw_size)

        # Grouping images
        with ZipFile(self._data_root).open('dataset.json') as f:
            labels = json.load(f)['labels']

        all_videos = {}
        for k, v in labels:
            video_name = osp.dirname(k)
            if video_name not in all_videos:
                all_videos[video_name] = []
            all_videos[video_name].append((k, v))

        all_videos_list = []
        for k, v in all_videos.items():
            np.random.shuffle(v)
            if len(v) > 1:
                all_videos_list.append((k, v))

        all_videos_list = sorted(all_videos_list, key=lambda x: x[0])
        self.all_videos_list = [x[1] for x in all_videos_list]

    def _get_zipfile(self):
        if self._zipfile is None:
            self._zipfile = ZipFile(self._data_root)
        return self._zipfile

    def _extract_from_labels(self, all_labels):
        images = {}
        for k, v in all_labels.items():
            if self._datatype == 'dir':
                images[k] = cv2.cvtColor(
                    cv2.imread(osp.join(self._data_root, v[0])),
                    cv2.COLOR_BGR2RGB
                )
            elif self._datatype == 'zip':
                with self._get_zipfile().open(v[0], 'r') as f:
                    if self.jpeg_decoder is not None and v[0].lower().endswith('.jpg') or v[0].lower().endswith('.jpeg'):
                        images[k] = self.jpeg_decoder.decode(f.read())
                    else:
                        image_array = np.frombuffer(f.read(), dtype=np.uint8)
                        images[k] = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            images[k] = cv2.cvtColor(images[k], cv2.COLOR_BGR2RGB)
        
        images = self.transform(**images)

        out = []
        for k, v in all_labels.items():
            data = {
                'image': images[k],
                'image_raw': F.interpolate(images[k].unsqueeze(0), (self.image_raw_size, self.image_raw_size)).squeeze(0),
                'cam2world': np.array(v[1][:16]).reshape(4, 4).astype(np.float32),
                'intrinsics': np.array(v[1][16:]).reshape(3, 3).astype(np.float32),
            }
            out.append(data)

        return out

    def __getitem__(self, group_idx):
        xs_data, xd_data = random.sample(self.all_videos_list[group_idx], k=2)

        xs, xd = self._extract_from_labels({'image': xs_data, 'xd_image': xd_data})
        xd['exp_image'] = xd['image']
        xd['mask'] = torch.ones_like(xd['image_raw'])
        out = {
            'data_name': 'celebvhq',
            'xs_data': xs,
            'xd_data': xd,
        }

        another_group_idx = group_idx
        while another_group_idx == group_idx:
            another_group_idx = random.randint(0, len(self) - 1)

        xs_aux_data, xd_aux_data = random.sample(self.all_videos_list[another_group_idx], 2)
        xs_aux, xd_aux = self._extract_from_labels({'image': xs_aux_data, 'xd_image': xd_aux_data})
        xd_aux['exp_image'] = xd_aux['image']
        xd_aux['mask'] = torch.ones_like(xd['image_raw'])

        out['xs_aux_data'] = xs_aux
        out['xd_aux_data'] = xd_aux

        return out

    def __len__(self):
        return len(self.all_videos_list)
