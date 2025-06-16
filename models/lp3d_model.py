import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_warn

import os
import os.path as osp
import fnmatch
from typing import Optional, Optional, List, Dict, OrderedDict, Tuple, Union
import cv2
import numpy as np
import lpips

from additional_modules.deeplabv3.deeplabv3 import DeepLabV3
from additional_modules.esrgan.rrdbnet_arch import RRDBNet
from additional_modules.eg3d.networks import OSGDecoder
from additional_modules.eg3d.eg3d_sampler import EG3DSampler
from additional_modules.eg3d.camera_utils import IntrinsicsSampler, LookAtPoseSampler
from additional_modules.projected_gan.discriminator import ProjectedDiscriminator
from additional_modules.segformer.backbone import Block, OverlapPatchEmbed
from losses.gan_loss import GANLoss
from losses.id_loss import IDLoss
from rendering.ray_sampler import RaySampler
from rendering.triplane_rendering.renderer import ImportanceRenderer
from utils.image_utils import tensor2img
from utils.registry import MODEL_REGISTRY


class PositionalEncoder(nn.Module):
    def __init__(self, img_size: int):
        """
            Architecture details: https://research.nvidia.com/labs/nxp/lp3d/media/paper.pdf
        """

        super().__init__()

        h_linspace = torch.linspace(-1, 1, img_size)
        w_linspace = torch.linspace(-1, 1, img_size)
        gh, gw = torch.meshgrid(h_linspace, w_linspace, indexing='xy')
        gh, gw = gh.unsqueeze(0), gw.unsqueeze(0)
        id_grid = torch.cat((gh, gw), dim=0).unsqueeze(0)
        self.register_buffer('id_grid', id_grid)

    def _add_positional_encoding(self, img: torch.Tensor) -> torch.Tensor:
        id_grid = self.id_grid.repeat(img.shape[0], 1, 1, 1)
        x = torch.cat((img, id_grid), dim=1)

        return x


class Lp3DELow(PositionalEncoder):
    def __init__(self, img_size: int = 512, img_channels: int = 3):
        """
            Architecture details: https://research.nvidia.com/labs/nxp/lp3d/media/paper.pdf
        """

        super().__init__(img_size)

        self.deeplabv3_backbone = DeepLabV3(input_channels=img_channels + 2)
        self.patch_embed = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2, in_chans=256, embed_dim=1024
        )

        self.block1 = Block(dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1)
        self.block2 = Block(dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1)
        self.block3 = Block(dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1)
        self.block4 = Block(dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1)
        self.block5 = Block(dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1)

        self.up1 = nn.PixelShuffle(upscale_factor=2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv1 = nn.Conv2d(256, 128, 3, 1, 1, bias=True)
        self.act1 = nn.ReLU()
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 96, 3, 1, 1, bias=True)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self._add_positional_encoding(img)

        x = self.deeplabv3_backbone(x)
        x, H, W = self.patch_embed(x)

        x = self.block1(x, H, W)
        x = self.block2(x, H, W)
        x = self.block3(x, H, W)
        x = self.block4(x, H, W)
        x = self.block5(x, H, W)

        x = x.reshape(img.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.up1(x)
        x = self.up2(x)

        x = self.conv1(x)
        x = self.act1(x)
        x = self.up3(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)

        return x


class Lp3DEHigh(PositionalEncoder):
    def __init__(self, img_size: int = 512, img_channels: int = 3):
        """
            Architecture details: https://research.nvidia.com/labs/nxp/lp3d/media/paper.pdf
        """

        super().__init__(img_size)

        self.conv1 = nn.Conv2d(img_channels + 2, 64, 7, 2, 3, bias=True)
        self.act1 = nn.LeakyReLU(0.01)

        self.conv2 = nn.Conv2d(64, 96, 3, 1, 1, bias=True)
        self.act2 = nn.LeakyReLU(0.01)

        self.conv3 = nn.Conv2d(96, 96, 3, 1, 1, bias=True)
        self.act3 = nn.LeakyReLU(0.01)

        self.conv4 = nn.Conv2d(96, 96, 3, 1, 1, bias=True)
        self.act4 = nn.LeakyReLU(0.01)

        self.conv5 = nn.Conv2d(96, 96, 3, 1, 1, bias=True)
        self.act5 = nn.LeakyReLU(0.01)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self._add_positional_encoding(img)
        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.act4(x)

        x = self.conv5(x)
        x = self.act5(x)

        return x


class Lp3DEncoder(nn.Module):
    def __init__(self, img_size: int = 512, img_channels: int = 3, triplane_nd: int = 32):
        """
            Architecture details: https://research.nvidia.com/labs/nxp/lp3d/media/paper.pdf
        """

        super().__init__()
        self.img_size = img_size

        self.elo = Lp3DELow(img_size, img_channels)
        self.ehi = Lp3DEHigh(img_size, img_channels)

        self.conv1 = nn.Conv2d(192, 256, 3, 1, 1, bias=True)
        self.act1 = nn.LeakyReLU(0.01)

        self.conv2 = nn.Conv2d(256, 128, 3, 1, 1, bias=True)
        self.act2 = nn.LeakyReLU(0.01)

        self.patch_embed = OverlapPatchEmbed(
            img_size=img_size // 2, patch_size=3, stride=2, in_chans=128, embed_dim=1024
        )
        self.transformer_block = Block(dim=1024, num_heads=2, mlp_ratio=2, sr_ratio=2)

        self.up = nn.PixelShuffle(upscale_factor=2)

        self.conv3 = nn.Conv2d(352, 256, 3, 1, 1, bias=True)
        self.act3 = nn.LeakyReLU(0.01)

        self.conv4 = nn.Conv2d(256, 128, 3, 1, 1, bias=True)
        self.act4 = nn.LeakyReLU(0.01)

        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.act5 = nn.LeakyReLU(0.01)

        self.conv6 = nn.Conv2d(128, triplane_nd * 3, 3, 1, 1, bias=True)

    def forward(self, img: torch.Tensor):
        assert img.shape[-1] == self.img_size and img.shape[-2] == self.img_size

        f_lo = self.elo(img)
        f_hi = self.ehi(img)

        f = torch.cat((f_lo, f_hi), dim=1)
        f = self.conv1(f)
        f = self.act1(f)

        f = self.conv2(f)
        f = self.act2(f)

        f, H, W = self.patch_embed(f)
        f = self.transformer_block(f, H, W)
        f = f.reshape(img.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()

        f = self.up(f)
        f = torch.cat((f, f_lo), dim=1)

        f = self.conv3(f)
        f = self.act3(f)

        f = self.conv4(f)
        f = self.act4(f)

        f = self.conv5(f)
        f = self.act5(f)

        f = self.conv6(f)

        return f


class Lp3DwRenderer(nn.Module):
    def __init__(self, neural_rendering_resolution, triplane_nd, **rendering_kwargs):
        super().__init__()

        self.triplane_nd = triplane_nd
        self.neural_rendering_resolution = neural_rendering_resolution
        self.rendering_kwargs = rendering_kwargs

        self.triplane_encoder = Lp3DEncoder(triplane_nd=triplane_nd)
        self.renderer = ImportanceRenderer()
        self.decoder = OSGDecoder(
            triplane_nd,
            {
                'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                'decoder_output_dim': triplane_nd
            }
        )
        self.ray_sampler = RaySampler()
        self.superresolution = RRDBNet()

    def render(
        self,
        planes: torch.Tensor,
        cam2world: torch.Tensor,
        intrinsics: torch.Tensor,
        upsample: bool = True,
        patch_indices: Optional[torch.Tensor] = None
    ):
        """
        Render the triplane using cam2wolrd and intrinsics matrices.
        Note that here we use ESRGAN for super-resolution and therefore the 29 last channels in the 
        feature image is redundant. You can replace the super-resolution with the one in EG3D but it
        is more inconsistent than ESRGAN in our experiment, though it is more cripsy.

        Parameters:
            - triplane (Tensor): Canonicalized triplane to render.
            - cam2world (Tensor): cam2world matrix.
            - intrinsics (Tensor): Camera intrinsics.
            - upsample (Bool): If True, use super-resolution.
            - patch_indices (Tensor): Only render pixels in patch_indices. If None, render the whole image.
        Returns (as dict):
            - image_raw (Tensor) [Range -1..1]: The rendered low-res images.
            - image (Tensor) [Range -1..1]: The super-resolved rendered images.
            - depth (Tensor): The depth image.
            - feature_image (Tensor): 32-channel feature images. Can be useful if you want to use EG3D superres.
            - ray_entropy (Tensor): Ray entropy for calculating the ray entropy loss.
            The below metadata can be useful to stack multiple renderings:
            - cam2world (Tensor): The input cam2world.
            - intrinsics (Tensor): The input intrinsics.
        """
        ray_origins, ray_directions = self.ray_sampler(
            cam2world, intrinsics, self.neural_rendering_resolution
        )

        if patch_indices is not None:
            ray_origins = ray_origins[:, patch_indices]
            ray_directions = ray_directions[:, patch_indices]

        batch_size = cam2world.shape[0]

        feature_samples, depth_samples, _, ray_entropy = self.renderer(
            planes,
            self.decoder,
            ray_origins,
            ray_directions,
            self.rendering_kwargs
        )  # channels last

        H = W = int(np.round(np.sqrt(feature_samples.shape[1])))

        feature_image = feature_samples.permute(0, 2, 1)
        feature_image = feature_image.reshape(batch_size, feature_samples.shape[-1], H, W).contiguous()
        rgb_image_raw = feature_image[:, :3]

        if upsample:
            rgb_image = self.superresolution(rgb_image_raw)
        else:
            rgb_image = None
       
        depth_image = depth_samples.permute(0, 2, 1).reshape(batch_size, 1, H, W).contiguous()

        return {
            'image_raw': rgb_image_raw,
            'image': rgb_image,
            'planes': planes,
            'depth': depth_image,
            'feature_image': feature_image,
            'ray_entropy': ray_entropy,
            'cam2world': cam2world,
            'intrinsics': intrinsics
        }

    def canonicalize(self, image: torch.Tensor) -> torch.Tensor:
        """
        Transform the input image to the canonicalized 3D space which represented by a triplane

        Parameters:
            - image (Tensor): Input image
        Returns:
            - triplanes (Tensor): The canonical representation of the input
        """

        image = (image + 1) / 2.  # Legacy issue :(
        triplanes = self.triplane_encoder(image)
        B = triplanes.shape[0]
        triplanes = triplanes.view(B, 3, self.triplane_nd, triplanes.shape[-2], triplanes.shape[-1]).contiguous()

        return triplanes

    def forward(
        self,
        source_image: torch.Tensor,
        all_cam2worlds: List[torch.Tensor],
        all_intrinsics: List[torch.Tensor],
        upsample: bool,
    ) -> List[torch.Tensor]:
        """ 
        Render the source image using camera parameters from the driver(s).

        This inference function supports multiple camera inputs which saves a lot of computation because
        the source is only canonicalized once. Can be useful for training or testing.

        Parameters:
            - source_image (Tensor): The source image.
            - all_cam2wolrds (List of tensors): List of all cam2world matrices (4x4).
            - all_intrinsics (List of tensors): List of all intrinsics matrices (4x4).
            - upsample (bool): If true, use super-resolution module.
        Returns:
            - all_out (List of tensors): Rendered images corresponding to the camera list.
        """

        assert len(all_cam2worlds) == len(all_intrinsics)

        xs_triplane = self.canonicalize(source_image)

        all_out = []
        for cam2world, intrinsics in zip(all_cam2worlds, all_intrinsics):
            instance_out = self.render(
                xs_triplane,
                cam2world, intrinsics,
                upsample=upsample
            )
            all_out.append(instance_out)

        return all_out


@MODEL_REGISTRY.register()
class Lp3DLightningModel(L.LightningModule):
    def __init__(
        self,
        neural_rendering_resolution: int,
        subsampling_ratio: float,
        triplane_nd: int,
        triplane_h: int,
        triplane_w: int,
        pretrained_path: str,
        eg3d_network_pkl: str,
        arcface_network_pkl: str,
        rendering_kwargs,
        superresolution_kwargs,
        loss_kwargs,
        training_kwargs,
        val_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.triplane_nd = triplane_nd
        self.triplane_h = triplane_h
        self.triplane_w = triplane_w
        self.neural_rendering_resolution = neural_rendering_resolution
        self.subsampling_ratio = subsampling_ratio
        self.eg3d_network_pkl = eg3d_network_pkl
        self.arcface_network_pkl = arcface_network_pkl
        self.superresolution_opt = superresolution_kwargs
        self.loss_opt = loss_kwargs
        self.training_opt = training_kwargs 
        self.val_opt = val_kwargs
        self.rendering_kwargs = rendering_kwargs

        self._setup_modules()
        self._load(pretrained_path)

        self._setup_losses()
        self._freeze_module(self.training_opt['frozen_components'])

        self._setup_validation()

        self.automatic_optimization = False

    def _setup_validation(self):
        camera_lookat_point = torch.tensor(self.val_opt['lookat_point']).float()
        yaw_range = self.val_opt['yaw_range']
        pitch_range = self.val_opt['pitch_range']
        num_keyframes = self.val_opt['num_keyframes']
        radius = self.val_opt['radius']

        cam2worlds = []
        for view_idx in range(num_keyframes):
            yaw_angle = 3.14/2 + yaw_range * np.sin(2 * 3.14 * view_idx / num_keyframes)
            pitch_angle = 3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * view_idx / num_keyframes)
            cam2worlds.append(
                LookAtPoseSampler.sample(
                    yaw_angle, pitch_angle, radius,
                    camera_lookat_point,
                    yaw_angle, pitch_angle, 0,
                )
            )
        
        intrinsics = IntrinsicsSampler.sample(
            18.837, 0.5,
            0, 0,
            batch_size=1
        )

        self.register_buffer('val_cam2worlds', torch.cat(cam2worlds, dim=0))
        self.register_buffer('val_intrinsics', intrinsics.repeat(self.val_cam2worlds.shape[0], 1, 1))

    def _setup_modules(self):
        # For now only support 512x512 input image and 256x256 triplane
        self.lp3d = Lp3DwRenderer(neural_rendering_resolution=self.neural_rendering_resolution, triplane_nd=self.triplane_nd, **self.rendering_kwargs)
        self.eg3d_sampler = EG3DSampler(self.eg3d_network_pkl).requires_grad_(False)
        self.disc = ProjectedDiscriminator(c_dim=0, diffaug=True, p_crop=False)

    def _setup_losses(self):
        self.lpips_loss = lpips.LPIPS(net='vgg').requires_grad_(False).eval()
        self.gan_loss = GANLoss(
            **self.loss_opt['GANLoss']['params']
        ).to(self.device)
        self.id_loss = IDLoss(self.arcface_network_pkl).requires_grad_(False)

    def _load(self, pretrained_path: Optional[str]):
        """
        Loading pretrained weights. Can load pytorch/lightning weights.

        Parameters:
        - pretrained_path (Optional[str]): Path of the pretrained model.
        """
        if pretrained_path is not None:
            state_dict_raw = torch.load(pretrained_path, map_location='cpu')
            if 'state_dict' in state_dict_raw:
                state_dict_raw = state_dict_raw['state_dict']
                self.load_state_dict(state_dict_raw, strict=True)
            else:
                self.lp3d.load_state_dict(state_dict_raw, strict=True)

    def _freeze_module(self, frozen_components: List[str]):
        """
        Freeze parameters. Use regex matching

        Parameters:
            - frozen_components (list of strings): patterns of target parameters.
        """
        matched_components = set()
        for k, v in self.named_parameters():
            for pattern in frozen_components:
                if fnmatch.fnmatch(k, pattern):
                    rank_zero_warn(f'{k} is freezed')
                    matched_components.add(pattern)
                    v.requires_grad_(False)

        for pattern in frozen_components:
            if pattern not in matched_components:
                raise ValueError(f'No parameter matched {pattern}')

    def _gen_patch_indices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Generate patch indices for patch rendering.
        The patch is a square of size patch_size = rendering_res * subsampling_ratio

        Parameters: None
        Returns:
            - selected_indices (Tensor): Batched indices of the positions in the patch.
            - x, y (Tensor): Top-left corner of the patch.
            - patch_size (int): Size of the patch
        """
        selected_indices = torch.zeros((
            self.neural_rendering_resolution, self.neural_rendering_resolution
        ))

        patch_size = int(self.neural_rendering_resolution * self.subsampling_ratio)
        x = torch.randint(self.neural_rendering_resolution - patch_size, (1, ))
        y = torch.randint(self.neural_rendering_resolution - patch_size, (1, ))
        selected_indices[x: x + patch_size, y: y + patch_size] = 1

        selected_indices = selected_indices.flatten().bool()
        return selected_indices, x, y, patch_size

    def apply_model(
        self,
        xs_data: Dict[str, torch.Tensor],
        all_xd_data: List[Dict[str, torch.Tensor]],
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        """ 
        Render the source image using camera parameters from the driver(s).

        This inference function supports multiple camera inputs. Can be useful when
        training in which the loss is calculated on multiple views of a single
        source image

        Parameters:
            - xs_data: The source's data. Must have 'image' key in it
            - all_xd_data: All drivers' data. Each of them must have 'cam2world' and 'intrinsics'
        Returns:
            - all_out (List of tensor): reconstructed source from the driver's views.
            - all_xd_data_cropped: Cropped xd. Will be used for calculating losses.
        """
        xs_triplane = self.lp3d.canonicalize(xs_data['image'])

        if self.subsampling_ratio < 1 and self.training:
            all_xd_data_cropped = []
            patch_indices, patch_x, patch_y, patch_size = self._gen_patch_indices()

            for xd_data in all_xd_data:
                xd_data_cropped = {k: v for k, v in xd_data.items()}
                xd_data_cropped['image_raw'] = xd_data_cropped['image_raw'][
                    :, :, patch_x: patch_x + patch_size, patch_y: patch_y + patch_size
                ]
                all_xd_data_cropped.append(xd_data_cropped)
        else:
            all_xd_data_cropped = all_xd_data
            patch_indices = None

        all_out = []
        for xd_data_cropped in all_xd_data_cropped:
            instance_out = self.lp3d.render(
                xs_triplane,
                xd_data_cropped['cam2world'], xd_data_cropped['intrinsics'],
                patch_indices=patch_indices,
                upsample=self.loss_opt['losses_on_hr'] or (not self.training)
            )
            all_out.append(instance_out)

        return all_out, all_xd_data_cropped

    def _calculate_paired_losses(self, target: torch.Tensor, gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate pixel-aligned losses between target and ground-truth.
        The losses include identity, perceptual, and L1 losses.
        """

        total_loss = 0
        loss_dict = {}

        # ID loss
        id_weight = self.loss_opt['id']['weight']
        if id_weight > 0:
            id_loss = self.id_loss(target, gt)
            total_loss += id_weight * id_loss
            loss_dict['id'] = id_loss

        # Perceptual loss
        lpips_weight = self.loss_opt['lpips']['weight']
        if lpips_weight > 0:
            lpips_loss = self.lpips_loss(target, gt).mean()
            total_loss += lpips_weight * lpips_loss
            loss_dict['lpips'] = lpips_loss

        # L1 loss 
        l1_weight = self.loss_opt['l1']['weight']
        if l1_weight > 0:
            l1_loss = F.l1_loss(target, gt)
            total_loss += l1_weight * l1_loss
            loss_dict['l1'] = l1_loss

        loss_dict['total'] = total_loss
        return loss_dict

    def _calculate_image_loss(
        self,
        target: Dict[str, torch.Tensor],
        gt: Dict[str, torch.Tensor],
        log_key: str
    ) -> torch.Tensor:
        """
        Calculate training losses. Also logs the losses.

        Parameters:
            - target: Prediction. Must have image_raw and optionally image if losses_on_hr is True
            - gt: Ground-truth. Must have image_raw and optionally image if losses_on_hr is True
        Returns:
            - losses: Total loss.
        """

        # Calculate losses on low res output
        lr_loss_dict = self._calculate_paired_losses(target['image_raw'], gt['image_raw'])

        loss = lr_loss_dict['total']
        for k, v in lr_loss_dict.items():
            if k != 'total':
                self.log(f"training_losses/{log_key}_lr/{k}", v)

        # Calculate losses on high res output
        if self.loss_opt['losses_on_hr']:
            hr_loss_dict = self._calculate_paired_losses(target['image'], gt['image'])
            loss += hr_loss_dict['total']
            for k, v in hr_loss_dict.items():
                if k != 'total':
                    self.log(f"training_losses/{log_key}_hr/{k}", v)
        
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        """
        Swap the dummy data with synthetic data from EG3D
        If using the dummy dataset, the model will be trained with pure
        synthetic data
        """
        num_dummies = batch['data_name'].count('dummy')
        if num_dummies == 0:
            return
        xs_data, xd_data = self.eg3d_sampler(num_views=2, batch_size=num_dummies)

        cnt = 0
        for idx, data_name in enumerate(batch['data_name']):
            if data_name == 'dummy':
                for k in xs_data:
                    batch['xs_data'][k][idx] = xs_data[k][cnt]
                    batch['xd_data'][k][idx] = xd_data[cnt]
                cnt += 1

    def training_step(self, batch, batch_idx):
        """
        Note that there is a bug in pytorch lightning that the global step is increased multiple time per step if
        there are more than one optimizer (which is the case if you use GAN loss).
        [This issue](https://github.com/Lightning-AI/pytorch-lightning/issues/17958)
        provides a quick fix but since the pytorch lightning team is working on it
        (https://github.com/Lightning-AI/pytorch-lightning/issues/17958) so I don't add the fix here.
        It doesn't affect anything except that your saved checkpoint's index is doubled.
        """
        gan_weight = self.loss_opt['gan']['weight']

        if gan_weight == 0.0:
            g_optimizer = self.optimizers()
        else:
            g_optimizer, d_optimizer = self.optimizers()

        xs_data = batch['xs_data']
        xd_data = batch['xd_data']

        # -------------------- Optimizing G ----------------------------
        self.toggle_optimizer(g_optimizer)

        # Use reference-view and multiple-view losses as suggested in Lp3D
        if batch_idx % self.training_opt['ref_loss_freq'] == 0:
            out, xd_data = self.apply_model(xs_data, [xs_data])[0]
            g_loss = self._calculate_image_loss(out, xd_data, 'ref')
        else:
            out, xd_data = self.apply_model(xs_data, [xd_data])[0]
            g_loss = self._calculate_image_loss(out, xd_data, 'mult')

        # Ray entropy loss
        ray_entropy_weight = self.loss_opt['ray_entropy']['weight']
        if  ray_entropy_weight > 0:
            g_loss += ray_entropy_weight * out['ray_entropy']
            self.log('training_losses/ray_entropy', out['ray_entropy'])

        # gan loss
        if gan_weight > 0:
            g_pred_fake = self.disc(out['image_raw'], c=None)
            g_gan_loss = self.gan_loss(g_pred_fake, True, dis_update=False)
            g_loss += gan_weight * g_gan_loss
            self.log('training_losses/g_gan', g_gan_loss)

        self.manual_backward(g_loss)
        g_optimizer.step()
        g_optimizer.zero_grad()
        self.untoggle_optimizer(g_optimizer)
        # -------------------- End of optimizing G ----------------------------

        if self.loss_opt['gan']['weight'] == 0:
            return

        # -------------------- optimizing D -----------------------------------
        self.toggle_optimizer(d_optimizer) 

        assert out['image_raw'].shape == xd_data['image_raw'].shape

        d_pred_fake = self.disc(out['image_raw'].detach(), c=None)
        d_pred_real = self.disc(xd_data['image_raw'], c=None)

        d_loss_fake = self.gan_loss(d_pred_fake, False, dis_update=True)
        d_loss_real = self.gan_loss(d_pred_real, True, dis_update=True)
        d_loss = (d_loss_fake + d_loss_real) / 2.0

        self.log('training_losses/d_fake', d_loss_fake)
        self.log('training_losses/d_real', d_loss_real)
        self.log('training_losses/d_all', d_loss)

        self.manual_backward(d_loss)
        d_optimizer.step()
        d_optimizer.zero_grad()
        self.untoggle_optimizer(d_optimizer)

    def generate_novel_views(self, batch) -> List[np.ndarray]:
        xs_data = batch['xs_data']
        batch_size = xs_data['image'].shape[0]
        assert batch_size == 1, "Only accept batch size of 1"

        xs_image = tensor2img(xs_data['image'][0], min_max=(-1, 1))

        cam_batch = self.val_opt['cam_batch']
        num_keyframes = self.val_intrinsics.shape[0]
        save_frames = []

        for cam_batch_idx in range(0, num_keyframes, cam_batch):
            xd_data = {
                'intrinsics': self.val_intrinsics[cam_batch_idx: cam_batch_idx + cam_batch],
                'cam2world': self.val_cam2worlds[cam_batch_idx: cam_batch_idx + cam_batch],
            }
            
            batch_xs_data = {
                'image': xs_data['image'].repeat(xd_data['intrinsics'].shape[0], 1, 1, 1)
            }

            out = self.apply_model(batch_xs_data, [xd_data,])[0]

            for instance_idx in range(out['image'].shape[0]):
                lr_frame = tensor2img(out['image_raw'][instance_idx], min_max=(-1, 1))
                if 'image' in out:
                    hr_frame = tensor2img(out['image'][instance_idx], min_max=(-1, 1))
                    lr_frame = cv2.resize(lr_frame, (hr_frame.shape[0], hr_frame.shape[1]))
                    vis = np.hstack((xs_image, lr_frame, hr_frame))
                else:
                    xs_image = cv2.resize(xs_image, (lr_frame.shape[0], lr_frame.shape[1]))
                    vis = np.hstack((xs_image, lr_frame))

                save_frames.append(vis)

        return save_frames

    def validation_step(self, batch, batch_idx):
        """
        For each validation input image, render novel views using a fixed camera trajectory
        """
        save_frames = self.generate_novel_views(batch)
        save_path = self.trainer.checkpoint_callback.dirpath
        save_name = osp.splitext(osp.basename(batch['xs_key'][0]))[0]
        save_path = osp.join(
            '/'.join(save_path.split('/')[:-1]),
            f'visualization/{self.global_step:08d}/{save_name}.avi'
        )
        os.makedirs(osp.dirname(save_path), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vid_out = cv2.VideoWriter(save_path, fourcc, 20, save_frames[-1].shape[:2][::-1])
        for frame in save_frames:
            vid_out.write(frame)

    def configure_optimizers(self):
        g_params = self.lp3d.parameters()
        g_params = filter(lambda x: x.requires_grad, g_params)

        g_optimizer = torch.optim.AdamW(
            g_params,
            lr=self.training_opt['learning_rate'],
            betas=self.training_opt['betas'],
            eps=self.training_opt['eps'],
            weight_decay=self.training_opt['weight_decay']
        )

        if self.loss_opt['gan']['weight'] == 0.0:
            return g_optimizer

        d_params = self.disc.parameters()
        d_optimizer = torch.optim.AdamW(
            d_params,
            lr=self.training_opt['learning_rate'],
            betas=self.training_opt['betas'],
            eps=self.training_opt['eps'],
            weight_decay=self.training_opt['weight_decay']
        )

        return g_optimizer, d_optimizer
