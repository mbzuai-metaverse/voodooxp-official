import torch

import click
import cv2
import glob
import os
import os.path as osp
from tqdm import tqdm
from typing import List, Optional, Union
import yaml
import numpy as np

from additional_modules.eg3d.camera_utils import IntrinsicsSampler, LookAtPoseSampler
from models.voodooxp_model import ExpLp3DwRenderer
from preprocessing.test_preprocess import TestDataPreprocessor
from utils.image_utils import tensor2img


IMAGE_EXTS = ['.jpeg', '.jpg', '.png', '.webp']
VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']


def tensor_from_path(img_path: str) -> torch.Tensor:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))[None, :, :, :] / 255.
    img = (img * 2 - 1)
    img = torch.from_numpy(img).float()

    return img


class Inferencer:
    def __init__(
        self,
        model_config_path,
        weight_path,
        skip_preprocess,
        undo_alignment,
        device,
        batch_size
    ):
        # Data preprocessing, including alignment, cropping, foreground extraction, and pose estimation
        self._device = device
        self._batch_size = batch_size
        self._skip_preprocess = skip_preprocess
        self._undo_alignment = undo_alignment
        self._preprocessor = TestDataPreprocessor(device)

        # Preparing model
        with open(model_config_path, 'r') as f:
            model_cfgs = yaml.safe_load(f)['model']['params']
        model = ExpLp3DwRenderer(
            model_cfgs['neural_rendering_resolution'],
            model_cfgs['triplane_nd'],
            use_aug=False,
            **model_cfgs['rendering_kwargs']
        ).to(device)

        state_dict = torch.load(weight_path, map_location='cpu')

        if 'state_dict' in state_dict:  # Load from pytorch lightning checkpoints
            state_dict_raw = state_dict['state_dict']
            state_dict_cleaned = {}
            for k, v in state_dict_raw.items():
                if k.startswith('lp3d.'):
                    state_dict_cleaned[k.replace('lp3d.', '', 1)] = v
        else:  # Load from cleaned checkpoints
            state_dict_cleaned = state_dict

        model.load_state_dict(state_dict_cleaned, strict=True)
        model.eval()
        self._model = model

    @torch.no_grad()
    def __call__(
        self,
        source_path: Union[str, List[str]],
        driver_path: Union[str, List[str]],
        save_path: Union[str, List[str]],
        custom_expression_path: Optional[Union[str, List[str]]] = None,
        custom_intrinsics: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        custom_cam2world: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ):

        # Support single or batched inputs
        all_source_paths = [source_path] if type(source_path) is str else source_path
        all_driver_paths = [driver_path] if type(driver_path) is str else driver_path
        all_save_paths = [save_path] if type(save_path) is str else save_path

        all_custom_expression_paths = \
            [custom_expression_path] if type(custom_expression_path) is str else custom_expression_path
        if all_custom_expression_paths is None:
            all_custom_expression_paths = [None] * len(all_source_paths)

        all_custom_intrinsics = \
            [custom_intrinsics] if type(custom_intrinsics) is torch.Tensor else custom_intrinsics
        all_custom_cam2worlds = \
            [custom_cam2world] if type(custom_cam2world) is torch.Tensor else custom_cam2world

        all_image_paths = all_source_paths + all_driver_paths + all_custom_expression_paths
        all_preprocessed_data = []

        for image_path in all_image_paths:
            if image_path is None:
                preprocessed_data = None
            elif not self._skip_preprocess:
                preprocessed_data = self._preprocessor.from_path(image_path, self._device, keep_bg=False)
            elif self._skip_preprocess:
                preprocessed_data = {'image': tensor_from_path(image_path).to(self._device)}

            all_preprocessed_data.append(preprocessed_data)
            
        all_source_data = all_preprocessed_data[: len(all_source_paths)]
        all_driver_data = all_preprocessed_data[len(all_source_paths): len(all_source_paths) + len(all_driver_paths)]
        all_custom_expression_data = all_preprocessed_data[-len(all_custom_expression_paths):]

        # Use expression image, intrinsics, and cam2world that are not extracted from the driver
        # If not provided, use the ones extracted from the driver
        for driver_data, custom_expression_data in zip(all_driver_data, all_custom_expression_data):
            if custom_expression_data is not None:
                driver_data['exp_image'] = custom_expression_data['image']
            else:
                driver_data['exp_image'] = driver_data['image']

        if all_custom_intrinsics is not None:
            for driver_data, custom_intrinsics in zip(all_driver_data, all_custom_intrinsics):
                driver_data['intrinsics'] = custom_intrinsics

        if all_custom_cam2worlds is not None:
            for driver_data, custom_cam2world in zip(all_driver_data, all_custom_cam2worlds):
                driver_data['cam2world'] = custom_cam2world

        num_batches = len(all_source_data) // self._batch_size
        for batch_start in range(0, len(all_source_data), self._batch_size):
            batch_source_data = all_source_data[batch_start: batch_start + self._batch_size]
            batch_driver_data = all_driver_data[batch_start: batch_start + self._batch_size]

            batch_out = self._model(
                source_image=torch.cat([source_data['image'] for source_data in batch_source_data]),
                exp_image=torch.cat([driver_data['exp_image'] for driver_data in batch_driver_data]),
                all_cam2worlds=[torch.cat([driver_data['cam2world'] for driver_data in batch_driver_data])],
                all_intrinsics=[torch.cat([driver_data['intrinsics'] for driver_data in batch_driver_data])],
                upsample=True,
                skip_aug=True
            )[0]

            for idx in range(len(batch_source_data)):
                pred_hr = tensor2img(batch_out['image'][idx], min_max=(-1, 1))
                # pred_lr = tensor2img(batch_out['image_raw'][idx], min_max=(-1, 1))
                # pred_lr = cv2.resize(pred_lr, pred_hr.shape[:2])

                if self._undo_alignment:
                    pred_hr = self._preprocessor.undo_alignment(pred_hr, batch_driver_data[idx]['crop_params'])
                    # pred_lr = self._preprocessor.undo_alignment(pred_lr, batch_driver_data[idx]['crop_params'])

                dst = all_save_paths[batch_start + idx]
                os.makedirs(osp.dirname(dst), exist_ok=True)
                cv2.imwrite(dst, pred_hr)


@torch.no_grad()
@click.command()
@click.option('--source_root', type=str, required=True, help='Source root')
@click.option('--driver_root', type=str, required=True, help='Source root')
@click.option('--model_config_path', type=str, required=True, help='Model config path')
@click.option(
    '--render_mode', type=click.Choice(['novel_view', 'driver_view', 'frontal_view']), required=True,
    help="""novel_view: Render image with driver's expression using a fixed camera trajectory
            driver_view: Render image with driver's expresison with driver's pose
         """
)
@click.option('--batch_size', type=int, default=1, help='Inference batch size')
@click.option('--weight_path', type=str, required=True, help='Pretrained weight path')
@click.option('--save_root', type=str, required=True, help='Save root')
@click.option('--skip_preprocess', is_flag=True, help='Do not use preprocessing')
@click.option('--pairwise', is_flag=True, help='Predict results for all pair of source and driver')
@click.option(
    '--undo_alignment', is_flag=True, help='Undo the EG3D alignment, paste the result back to the original crop'
)
def main(
    source_root,
    driver_root,
    model_config_path,
    render_mode,
    batch_size,
    weight_path,
    save_root,
    skip_preprocess,
    pairwise,
    undo_alignment
):
    '''
    Inference LP3D model. For each source image, render its novel views using a fixed camera trajectory
    '''

    assert render_mode != 'driver_view' or (not skip_preprocess), "driver_view mode requires preprocessing to calculate pose"

    # Preparing data
    device = 'cuda'

    inferencer = Inferencer(
        model_config_path=model_config_path,
        weight_path=weight_path,
        skip_preprocess=skip_preprocess,
        undo_alignment=undo_alignment,
        batch_size=batch_size,
        device=device
    )

    # Scan images
    if osp.isfile(source_root):
        source_paths = [source_root]
    else:
        source_paths = sorted(glob.glob(osp.join(source_root, '*')))
        source_paths = [x for x in source_paths if osp.splitext(x.lower())[-1] in IMAGE_EXTS]

    if osp.isfile(driver_root):
        driver_paths = [driver_root]
    else:
        driver_paths = sorted(glob.glob(osp.join(driver_root, '*')))
        driver_paths = [x for x in driver_paths if osp.splitext(x.lower())[-1] in IMAGE_EXTS]

    # For novel view synthesis
    num_keyframes = 200
    yaw_range = 0.4
    pitch_range = 0.3
    rotating_cam2worlds = []
    camera_lookat_point = torch.tensor([0, 0, 0.2]).float().to(device)
    z = torch.randn(1, 512).to(device)
    for view_idx in range(num_keyframes):
        yaw_angle = 3.14/2 + yaw_range * np.sin(2 * 3.14 * view_idx / num_keyframes)
        pitch_angle = 3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * view_idx / num_keyframes)
        rotating_cam2worlds.append(
            LookAtPoseSampler.sample(
                yaw_angle, pitch_angle, 2.7,
                camera_lookat_point,
                yaw_angle, pitch_angle, 0,
                device=device
            )
        )
    rotating_intrinsics = IntrinsicsSampler.sample(
        18.837, 0.5,
        0, 0,
        batch_size=1,
        device=device
    )

    canonical_cam2world = LookAtPoseSampler.sample(
        np.pi / 2, np.pi / 2, 2.7,
        camera_lookat_point,
        np.pi / 2, np.pi / 2, 0,
        device=device
    )
    canonical_intrinsics = rotating_intrinsics

    # Inference
    inference_source_paths = []
    inference_driver_paths = []
    inference_save_paths = []
    inference_intrinsics = []
    inference_cam2worlds = []

    if pairwise:
        for source_path in source_paths:
            for driver_path in driver_paths:
                source_name = osp.splitext(osp.basename(source_path))[0]
                driver_name = osp.splitext(osp.basename(driver_path))[0]
                save_path = osp.join(save_root, source_name, driver_name + '.png')

                inference_source_paths.append(source_path)
                inference_driver_paths.append(driver_path)
                inference_save_paths.append(save_path)

        inference_intrinsics = None
        inference_cam2worlds = None
    else:
        if len(source_paths) == 1:
            source_paths *= len(driver_paths)
        assert len(source_paths) == len(driver_paths), "Number of sources and drivers must be the same!"

        for driver_idx, (source_path, driver_path) in enumerate(zip(source_paths, driver_paths)):
            source_name = osp.splitext(osp.basename(source_path))[0]
            driver_name = osp.splitext(osp.basename(driver_path))[0]
            save_path = osp.join(save_root, source_name + '_' + driver_name + '.png')

            inference_source_paths.append(source_path)
            inference_driver_paths.append(driver_path)
            inference_save_paths.append(save_path)

            if render_mode == 'novel_view':
                inference_intrinsics.append(rotating_intrinsics)
                inference_cam2worlds.append(rotating_cam2worlds[driver_idx % num_keyframes])
            elif render_mode == 'frontal_view':
                inference_intrinsics.append(canonical_intrinsics)
                inference_cam2worlds.append(canonical_cam2world)

    if render_mode == 'driver_view':
        inference_intrinsics = None
        inference_cam2worlds = None

    inferencer(
        inference_source_paths,
        inference_driver_paths,
        inference_save_paths,
        None,
        inference_intrinsics,
        inference_cam2worlds,
    )


if __name__ == '__main__':
    main()
