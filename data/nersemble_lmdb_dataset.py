import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import numpy as np
import pickle as pkl
import lmdb
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
class NersembleLmdbDataset(Dataset):
    def __init__(
        self,
        data_root,
        mode,
        image_raw_size=128
    ):
        super().__init__()

        self._mode = mode
        assert self._mode in ['3dreconstruction', 'reenactment']

        self.image_raw_size = image_raw_size

        self._data_root = data_root

        if (not self._data_root.endswith('.lmdb')):
            raise ValueError(f'Data root is not a valid lmdb database')

        if TurboJPEG is not None:
            self.jpeg_decoder = TurboJPEG()
        else:
            self.jpeg_decoder = None

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

        # Grouping images
        self.frame_groups_recon = {}
        self.frame_groups_reenact = {}

        with open(osp.join(self._data_root, 'all_keys.pkl'), 'rb') as f:
            all_keys = pkl.load(f)

        for k in all_keys:
            group_name_recon = self._get_group_from_path(k, '3dreconstruction')
            group_name_reenact = self._get_group_from_path(k, 'reenactment')

            if '246' in group_name_recon:
                continue

            if group_name_recon not in self.frame_groups_recon:
                self.frame_groups_recon[group_name_recon] = []
            self.frame_groups_recon[group_name_recon].append(k)

            if group_name_reenact not in self.frame_groups_reenact:
                self.frame_groups_reenact[group_name_reenact] = []
            self.frame_groups_reenact[group_name_reenact].append(k)

        # Only keep groups with 2 or more elements
        self.group_names = []
        if mode == '3dreconstruction':
            for k, v in self.frame_groups_recon.items():
                if len(v) > 1:
                    self.group_names.append(k)
        else:
            for k, v in self.frame_groups_reenact.items():
                if len(v) > 1:
                    self.group_names.append(k)

        # Zip data placeholder
        self._lmdb_database = None

    @staticmethod
    def _get_group_from_path(path, mode):
        identity_idx, category_idx, cam_idx, frame_idx = path.split('/')
        if mode == '3dreconstruction':
            return '_'.join((identity_idx, category_idx, frame_idx))
        else:
           return '_'.join((identity_idx, cam_idx))

    def _get_lmdb_database(self):
        if self._lmdb_database is None:
            self._lmdb_database = lmdb.open(self._data_root, readonly=True)
        return self._lmdb_database.begin()

    def _extract_from_labels(self, all_labels):
        images = {}
        for k, v in all_labels.items():
            with self._get_lmdb_database() as txn:
                f = txn.get(v.encode())
                if self.jpeg_decoder is not None:
                    images[k] = self.jpeg_decoder.decode(f)
                else:
                    image_array = np.frombuffer(f, dtype=np.uint8)
                    images[k] = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            images[k] = cv2.cvtColor(images[k], cv2.COLOR_BGR2RGB)
        
        images = self.transform(**images)

        out = []
        for k, v in all_labels.items():
            label_key = '_label_' + v
            with self._get_lmdb_database() as txn:
                label_content = list(map(float, txn.get(label_key.encode()).decode().split(' ')))
            data = {
                'image': images[k],
                'image_raw': F.interpolate(
                    images[k].unsqueeze(0), (self.image_raw_size, self.image_raw_size)
                ).squeeze(0),
                'cam2world': np.array(label_content[:16]).reshape(4, 4).astype(np.float32),
                'intrinsics': np.array(label_content[16:]).reshape(3, 3).astype(np.float32),
            }
            out.append(data)

        return out

    def __getitem__(self, group_idx):
        group_name = self.group_names[group_idx]

        if self._mode == '3dreconstruction':
            xs_data, xd_data = random.choices(self.frame_groups_recon[group_name], k=2)
            xs, xd = self._extract_from_labels({'image': xs_data, 'xd_image': xd_data})
        else:
            while True:
                xs_data, xd_data = random.sample(self.frame_groups_reenact[group_name], 2)
                if ('GLASSES' in xs_data) and ('GLASSES' not in xd_data):
                    continue
                if ('GLASSES' not in xs_data) and ('GLASSES' in xd_data):
                    continue
                break
            xd_recon_group_name = self._get_group_from_path(xd_data, '3dreconstruction')
            xd_exp_data, = random.sample(self.frame_groups_recon[xd_recon_group_name], 1)
            xs, xd = self._extract_from_labels({'image': xs_data, 'xd_image': xd_data})
            xd_exp, = self._extract_from_labels({'image': xd_exp_data})
            xd['exp_image'] = F.interpolate(F.interpolate(xd_exp['image'].unsqueeze(0), (128, 128)), (512, 512)).squeeze(0)
            xd['mask'] = torch.ones_like(xd['image_raw'])

        out = {
            'data_name': 'nersemble',
            'xs_data': xs,
            'xd_data': xd,
        }

        if self._mode == 'reenactment':
            another_group_name = group_name
            while another_group_name == group_name:
                another_group_name = random.choice(self.group_names)
            xs_aux_data, xd_aux_data = random.sample(self.frame_groups_reenact[another_group_name], 2)

            xs_aux, xd_aux = self._extract_from_labels({'image': xs_aux_data, 'xd_image': xd_aux_data})
            xd_aux['exp_image'] = xd_aux['image']
            xd_aux['mask'] = torch.ones_like(xd['image_raw'])

            out['xs_aux_data'] = xs_aux
            out['xd_aux_data'] = xd_aux

        return out

    def __len__(self):
        return len(self.group_names)
