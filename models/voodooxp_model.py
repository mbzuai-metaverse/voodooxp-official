# https://youtu.be/1F3OGIFnW1k?si=Gxu43L2ZrISDnhTT
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import cv2
from kornia.augmentation import ColorJiggle, AugmentationSequential
from kornia.augmentation import CenterCrop
from kornia.augmentation import RandomChannelShuffle
import os
import numpy as np
import timm
from typing import Dict, Optional, List

from additional_modules.eg3d.camera_utils import IntrinsicsSampler, LookAtPoseSampler
from additional_modules.segformer.backbone import Block, OverlapPatchEmbed
from additional_modules.stablediffusion.attention import BasicTransformerBlock
from additional_modules.projected_gan.discriminator import ProjectedDiscriminator
from models.lp3d_model import Lp3DLightningModel, Lp3DELow, Lp3DEHigh, Lp3DwRenderer
from models.neutralizer_model import NeutralizerwRenderer
from utils.image_utils import tensor2img
from utils.registry import MODEL_REGISTRY


class DriverAugmentor(nn.Module):
    def __init__(self, input_resolution):
        super().__init__()

        self.color_jitter = AugmentationSequential(
            ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.),
            RandomChannelShuffle(),
        )

        self.input_resolution = input_resolution
        self.crop = CenterCrop(self.input_resolution // 2)

    @torch.no_grad()
    def __call__(self, faces, apply_color_aug=True):
        faces = faces.clone()

        faces = self.crop(faces)
        if apply_color_aug:
            faces = (faces + 1) / 2
            faces = self.color_jitter(faces)
            faces = faces * 2 - 1

        return faces


class ExpELow(Lp3DELow):
    def __init__(self, img_size: int = 512, img_channels: int = 3):
        super().__init__(img_size, img_channels)
        self.cross_attn1 = BasicTransformerBlock(dim=1024, n_heads=4, d_head=512, context_dim=384)
        self.cross_attn2 = BasicTransformerBlock(dim=1024, n_heads=4, d_head=512, context_dim=384)
        self.cross_attn3 = BasicTransformerBlock(dim=1024, n_heads=4, d_head=512, context_dim=384)
        self.cross_attn4 = BasicTransformerBlock(dim=1024, n_heads=4, d_head=512, context_dim=384)
        self.cross_attn5 = BasicTransformerBlock(dim=1024, n_heads=4, d_head=512, context_dim=384)

    def forward(self, img: torch.Tensor, exp_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self._add_positional_encoding(img)
        x = self.deeplabv3_backbone(x)
        x, H, W = self.patch_embed(x)
        x = self.block1(x, H, W)

        if exp_feat is not None:
            x = self.cross_attn1(x, exp_feat)

        x = self.block2(x, H, W)
        if exp_feat is not None:
            x = self.cross_attn2(x, exp_feat)

        x = self.block3(x, H, W)
        if exp_feat is not None:
            x = self.cross_attn3(x, exp_feat)

        x = self.block4(x, H, W)
        if exp_feat is not None:
            x = self.cross_attn4(x, exp_feat)

        x = self.block5(x, H, W)
        if exp_feat is not None:
            x = self.cross_attn5(x, exp_feat)

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


class ExpLp3DEncoder(nn.Module):
    def __init__(self, img_size: int = 512, img_channels: int = 3, triplane_nd: int = 32):
        super().__init__()
        self.img_size = img_size

        self.elo = ExpELow(img_size, img_channels)
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

    def forward(
        self, img: torch.Tensor,
        exp_feat: Optional[torch.Tensor] = None,
    ):
        assert img.shape[-1] == self.img_size and img.shape[-2] == self.img_size

        f_lo = self.elo(img, exp_feat)
        f_hi = self.ehi(img)

        # Caching
        self.f_hi = f_hi

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


class MLPMixer(nn.Module):
    def __init__(self, num_in_tokens, num_out_tokens):
        super().__init__()

        self.mixer = nn.Linear(num_in_tokens, num_out_tokens)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.mixer(x)
        x = x.permute(0, 2, 1)
        return x


class ExpLp3DwRenderer(Lp3DwRenderer):
    def __init__(self, neural_rendering_resolution, triplane_nd, use_aug, **rendering_kwargs):
        super().__init__(neural_rendering_resolution, triplane_nd, **rendering_kwargs)
        self.use_aug = use_aug

        del self.triplane_encoder
        torch.cuda.empty_cache()
        self.triplane_encoder = ExpLp3DEncoder(triplane_nd=self.triplane_nd)

        # Expression encoder
        num_exp_tokens = 40
        exp_token_dim = 384

        self.expression_aug = DriverAugmentor(self.neural_rendering_resolution)
        dino = timm.create_model('vit_small_patch8_224.dino', pretrained=True, num_classes=0)
        self.expression_encoder = nn.Sequential(OrderedDict([
            ('vit_backbone', dino),
            ('token_reducer', MLPMixer(dino.patch_embed.num_patches + 1, num_exp_tokens)),
            ('channel_reducer', nn.Linear(dino.embed_dim, exp_token_dim)),
        ]))

    def estimate_expression(self, exp_img: torch.Tensor, skip_aug: bool = False):
        exp_img = F.interpolate(exp_img, (256, 256))
        exp_img_dino = self.expression_aug(
            exp_img,
            apply_color_aug=self.training and self.use_aug and (not skip_aug),
        )

        dino_img_resolution = self.expression_encoder.vit_backbone.patch_embed.img_size[0]
        exp_img_dino = F.interpolate(exp_img_dino, (dino_img_resolution,) * 2)
        exp_feat = self.expression_encoder.vit_backbone.forward_features(exp_img_dino)
        exp_feat = self.expression_encoder.token_reducer(exp_feat)
        exp_feat = self.expression_encoder.channel_reducer(exp_feat)

        return exp_feat

    def canonicalize(self, image: torch.Tensor, exp_feat: torch.Tensor) -> torch.Tensor:
        """
        Transform the input image to the canonicalized 3D space which represented by a triplane. Also change
        the expression to the expression represented by exp_feat.

        Parameters:
            - image (Tensor): Input image.
            - exp_feat (Tensor): Expression representative vector.
        Returns:
            - triplanes (Tensor): The canonical representation of the input with the new expression.
        """

        image = (image + 1) / 2.  # Legacy issue :(
        triplanes = self.triplane_encoder(image, exp_feat)
        B = triplanes.shape[0]
        triplanes = triplanes.view(B, 3, self.triplane_nd, triplanes.shape[-2], triplanes.shape[-1]).contiguous()

        return triplanes

    def forward(
        self,
        source_image: torch.Tensor,
        exp_image: torch.Tensor,
        all_cam2worlds: List[torch.Tensor],
        all_intrinsics: List[torch.Tensor],
        upsample: bool,
        skip_aug: bool
    ) -> List[torch.Tensor]:
        """ 
        Transfer the expression of exp_image to the source and render it using camera parameters from the driver(s).

        This inference function supports multiple camera inputs which saves a lot of computation because
        the source is only canonicalized once. Can be useful for training or testing.

        Parameters:
            - xs_data (Tensor): The source image.
            - epx_image (Tensor): The image of the target expression.
            - all_cam2wolrds (List of tensors): List of all cam2world matrices (4x4).
            - all_intrinsics (List of tensors): List of all intrinsics matrices (4x4).
            - upsample (bool): If true, use super-resolution module.
            - skip_aug (bool): If true, not use the driver augmentation. should be set to true during inference.
        Returns:
            - all_out (List of tensors): Rendered images corresponding to the camera list.
        """

        assert len(all_cam2worlds) == len(all_intrinsics)

        exp_feat = self.estimate_expression(exp_image)
        xs_triplane_wNewExp = self.canonicalize(source_image, exp_feat)

        all_out = []
        for cam2world, intrinsics in zip(all_cam2worlds, all_intrinsics):
            instance_out = self.render(
                xs_triplane_wNewExp,
                cam2world, intrinsics,
                upsample=upsample
            )
            all_out.append(instance_out)

        return all_out


@MODEL_REGISTRY.register()
class VoodooXPLightningModel(Lp3DLightningModel):
    def __init__(
        self,
        neural_rendering_resolution: int,  # Render at this resolution and use superres to upsample to 512x512
        subsampling_ratio: float,
        triplane_nd: int,  # Triplane's number of channels
        triplane_h: int,  # Triplane height
        triplane_w: int,  # Triplane width
        pretrained_path: str,
        eg3d_network_pkl: str,
        arcface_network_pkl: str,
        neutralizer_network_pkl: str,
        use_aug: bool,
        rendering_kwargs,
        superresolution_kwargs,
        loss_kwargs,
        training_kwargs,
        val_kwargs
    ):
        self.use_aug = use_aug
        self.use_aug = use_aug
        self.neutralizer_network_pkl = neutralizer_network_pkl

        super().__init__(
            neural_rendering_resolution,
            subsampling_ratio,
            triplane_nd,
            triplane_h,
            triplane_w,
            pretrained_path,
            eg3d_network_pkl,
            arcface_network_pkl,
            rendering_kwargs,
            superresolution_kwargs,
            loss_kwargs,
            training_kwargs,
            val_kwargs
        )

        # This is used to frontalize faces
        lookat_point = torch.tensor(rendering_kwargs['lookat_point']).unsqueeze(0).float()
        canonical_cam2world = LookAtPoseSampler.sample(
            np.pi / 2, np.pi / 2, rendering_kwargs['camera_radius'],
            lookat_point,
            np.pi / 2, np.pi / 2, 0.0,
            batch_size=1
        )
        canonical_intrinsics = IntrinsicsSampler.sample(
            18.837, 0.5,
            0, 0,
            batch_size=1
        )

        self.register_buffer('lookat_point', lookat_point)
        self.register_buffer('canonical_cam2world', canonical_cam2world)
        self.register_buffer('canonical_intrinsics', canonical_intrinsics)

    def _setup_modules(self):
        # For now only support 512x512 input image and 256x256 triplane
        self.lp3d = ExpLp3DwRenderer(
            neural_rendering_resolution=self.neural_rendering_resolution,
            triplane_nd=self.triplane_nd,
            use_aug=self.use_aug,
            **self.rendering_kwargs
        )
        self.disc = ProjectedDiscriminator(c_dim=0, diffaug=True, p_crop=False)

    def _setup_losses(self):
        super()._setup_losses()

        self.neutralizer = NeutralizerwRenderer(128, self.triplane_nd, **self.rendering_kwargs).requires_grad_(False)
        neutralizer_state_dict = torch.load(self.neutralizer_network_pkl, map_location='cpu')
        self.neutralizer.load_state_dict(neutralizer_state_dict, strict=True)

    def apply_model(
        self,
        xs_data: Dict[str, torch.Tensor],
        xd_data: Dict[str, torch.Tensor],
    ):
        """
        Reenact the source image using driver(s).

        This inference function support multiple camera inputs. Can be useful when
        training in which the loss is calculated on multiple views of a single
        source image

        Parameters:
            - xs_data: The source's data. Must have 'image' key in it
            - all_xds_data: All drivers' data. Each of them must have 'image, 'cam2world', and 'intrinsics'
            - calc_cross_reenactment: Use drivers cyclically shifted by 1 for cross-reenactment
            - calc_source_frontal: Calculate source frontal face
        """
        # Calculate driver expression
        xd_exp_feat = self.lp3d.estimate_expression(xd_data['exp_image'])
        xs_triplane_newExp = self.lp3d.canonicalize(xs_data['image'], xd_exp_feat)

        if self.subsampling_ratio < 1 and self.training:
            patch_indices, patch_x, patch_y, patch_size = self._gen_patch_indices()
            if 'image_raw' in xd_data:
                xd_data['image_raw_cropped'] = xd_data['image_raw'][
                    :, :, patch_x: patch_x + patch_size, patch_y: patch_y + patch_size
                ]
            if 'mask' in xd_data:
                xd_data['mask_cropped'] = xd_data['mask'][
                    :, :, patch_x: patch_x + patch_size, patch_y: patch_y + patch_size
                ]
        else:
            patch_indices = None

        out = self.lp3d.render(
            xs_triplane_newExp, xd_data['cam2world'], xd_data['intrinsics'], upsample=not self.training,
            patch_indices=patch_indices
        )
        if self.subsampling_ratio < 1:
            out['image_raw_cropped'] = out['image_raw']
            out.pop('image_raw')

        return out

    def _extract_eyes(self, planes):
        canonical_cam2world = self.canonical_cam2world.repeat(planes.shape[0], 1, 1)
        canonical_intrinsics = self.canonical_intrinsics.repeat(planes.shape[0], 1, 1)

        selected_indices_left = torch.zeros((
            self.neural_rendering_resolution, self.neural_rendering_resolution
        ))
        selected_indices_left[82: 110, 82: 110] = 1
        selected_indices_left = selected_indices_left.flatten().bool()
        left_eye = self.render(
            planes, canonical_cam2world, canonical_intrinsics, upsample=False,
            patch_indices=selected_indices_left
        )['image_raw']

        selected_indices_right = torch.zeros((
            self.neural_rendering_resolution, self.neural_rendering_resolution
        ))
        selected_indices_right[82: 110, 152: 180] = 1
        selected_indices_right = selected_indices_right.flatten().bool()
        right_eye = self.render(
            planes, canonical_cam2world, canonical_intrinsics, upsample=False,
            patch_indices=selected_indices_right
        )['image_raw']

        return left_eye, right_eye

    def _render_face_only(self, planes):
        canonical_cam2world = self.canonical_cam2world.repeat(planes.shape[0], 1, 1)
        canonical_intrinsics = self.canonical_intrinsics.repeat(planes.shape[0], 1, 1)

        selected_indices_left = torch.zeros((
            self.neural_rendering_resolution, self.neural_rendering_resolution
        ))
        selected_indices_left[
            self.neural_rendering_resolution // 4: self.neural_rendering_resolution // 4 * 3,
            self.neural_rendering_resolution // 4: self.neural_rendering_resolution // 4 * 3,
        ] = 1
        selected_indices_left = selected_indices_left.flatten().bool()
        face = self.voodooxp.render(
            planes, canonical_cam2world, canonical_intrinsics, upsample=False,
            patch_indices=selected_indices_left
        )['image_raw']

        return face

    def _eyegaze_loss(self, pred_planes, gt_planes):
        pred_left_eye, pred_right_eye = self._extract_eyes(pred_planes)
        gt_left_eye, gt_right_eye = self._extract_eyes(gt_planes)

        left_eye_loss = F.l1_loss(pred_left_eye, gt_left_eye, reduce=False)
        right_eye_loss = F.l1_loss(pred_right_eye, gt_right_eye, reduce=False)
        eye_loss = (left_eye_loss + right_eye_loss).mean(dim=(1, 2, 3))
        return eye_loss

    def _calculate_image_loss(self, target, gt, log_key):
        lr_loss_dict = self._calculate_paired_losses(
            target['image_raw_cropped'],
            gt['image_raw_cropped']
        )

        loss = lr_loss_dict['total']
        for k, v in lr_loss_dict.items():
            if k != 'total':
                self.log(f"training_losses/{log_key}_lr/{k}", v)

        return loss

    def training_step(self, batch, batch_idx):
        gan_weight = self.loss_opt['gan']['weight']

        if gan_weight == 0.0:
            g_optimizer = self.optimizers()
        else:
            g_optimizer, d_optimizer = self.optimizers()

        xs1_data = batch['xs_data']
        xd1_data = batch['xd_data']
        xs2_data = batch['xs_aux_data']
        xd2_data = batch['xd_aux_data']

        # -------------------- Optimizing G ----------------------------
        self.toggle_optimizer(g_optimizer)

        # self-reenactment losses
        self_out = self.apply_model(
            xs1_data,
            xd1_data,
        )

        g_loss = self._calculate_image_loss(self_out, xd1_data, 'self')

        # For debugging
        if 'DEBUG_INPUTS' in os.environ and os.environ['DEBUG_INPUTS'] == '1':
            debug_inputs = True
        else:
            debug_inputs = False

        if debug_inputs:
            img1 = tensor2img(xs1_data['image'], min_max=(-1, 1))
            img2 = tensor2img(xd1_data['image'], min_max=(-1, 1))
            img3 = tensor2img(xs2_data['image'], min_max=(-1, 1))
            img4 = tensor2img(xd2_data['image'], min_max=(-1, 1))
            cv2.imwrite('debug/all_inputs.png', np.hstack((img1, img2, img3, img4)))

            img1 = tensor2img(self_out['image_raw_cropped'], min_max=(-1, 1))
            img2 = tensor2img(xd1_data['image_raw_cropped'], min_max=(-1, 1))
            cv2.imwrite('debug/self_loss_input.png', np.hstack((img1, img2)))

        # cross-reenactment losses
        neutralizing_loss_weight = self.loss_opt['neutralizing_loss']['weight']
        if neutralizing_loss_weight > 0:
            cross_out = self.apply_model(xs1_data, xd2_data)

            xs1d2_full_size = self.lp3d.render(
                cross_out['planes'], xs1_data['cam2world'], xs1_data['intrinsics'], upsample=False
            )['image_raw']

            fake_samples = cross_out['image_raw_cropped']
            xd1_planes = self.neutralizer.canonicalize(F.interpolate(xd1_data['image_raw'], scale_factor=2))
            cross_planes = self.neutralizer.canonicalize(F.interpolate(xs1d2_full_size, scale_factor=2))
            neutralizing_loss = F.l1_loss(xd1_planes, cross_planes)
            self.log('cross/neu', neutralizing_loss)
            g_loss += neutralizing_loss_weight * neutralizing_loss

            if debug_inputs:
                img1 = tensor2img(xs1d2_full_size, min_max=(-1, 1))
                canonical_cam2world = self.canonical_cam2world.repeat(xs1d2_full_size.shape[0], 1, 1)
                canonical_intrinsics = self.canonical_intrinsics.repeat(xs1d2_full_size.shape[0], 1, 1)
                with torch.no_grad():
                    img2 = self.neutralizer.render(
                        cross_planes, canonical_cam2world, canonical_intrinsics, upsample=False
                    )['image_raw']
                    img3 = self.neutralizer.render(
                        xd1_planes, canonical_cam2world, canonical_intrinsics, upsample=False
                    )['image_raw']
                img2 = cv2.resize(tensor2img(img2, min_max=(-1, 1)), (256, 256))
                img3 = cv2.resize(tensor2img(img3, min_max=(-1, 1)), (256, 256))
                cv2.imwrite('debug/neutralizer.png', np.hstack((img1, img2, img3)))

            cycle_consistency_weight = self.loss_opt['cycle_consistency']['weight']
            if cycle_consistency_weight > 0:
                cycle_driver = {
                    'exp_image': xs1d2_full_size,
                    'image_raw': xd2_data['image_raw'],
                    'intrinsics': xd2_data['intrinsics'],
                    'cam2world': xd2_data['cam2world']
                }
                x_s2_s1_d2 = self.apply_model(xs2_data, cycle_driver)
                cycle_consistency_loss = F.l1_loss(
                    x_s2_s1_d2['image_raw_cropped'],
                    cycle_driver['image_raw_cropped']
                )
                self.log('cross/cyc', cycle_consistency_loss)
                g_loss += cycle_consistency_weight * cycle_consistency_loss

                if debug_inputs:
                    img1 = tensor2img(x_s2_s1_d2['image_raw_cropped'], min_max=(-1, 1))
                    img2 = tensor2img(cycle_driver['image_raw_cropped'], min_max=(-1, 1))
                    cv2.imwrite('debug/cyc_loss.png', np.hstack((img1, img2)))
        else:
            fake_samples = self_out['image_raw']

        # gan loss
        if gan_weight > 0:
            g_pred_fake = self.disc(fake_samples, c=None)
            g_gan_loss = self.gan_loss(g_pred_fake, True, dis_update=False)
            g_loss += gan_weight * g_gan_loss
            self.log('training_losses/g_gan', g_gan_loss)

        # # eye gaze loss
        # eyegaze_weight = self.loss_opt['eyegaze']['weight']
        # if eyegaze_weight > 0:
        #     eye_loss = (self._eyegaze_loss(self_out['planes'], driver_planes) * batch['eye_weight']).mean()
        #     g_loss += eyegaze_weight * eye_loss
        #     self.log(f'training_losses/eyegaze_loss', eye_loss)

        self.manual_backward(g_loss)
        g_optimizer.step()
        g_optimizer.zero_grad()
        self.untoggle_optimizer(g_optimizer)
        # -------------------- End of optimizing G ----------------------------

        if self.loss_opt['gan']['weight'] == 0:
            return

        # -------------------- optimizing D -----------------------------------
        self.toggle_optimizer(d_optimizer)

        assert fake_samples.shape == xd1_data['image_raw_cropped'].shape

        d_pred_fake = self.disc(fake_samples.detach(), c=None)
        # d_pred_real = self.disc(batch['ffhq_data']['image_raw'], c=None)
        d_pred_real = self.disc(xd1_data['image_raw_cropped'], c=None)

        d_loss_fake = self.gan_loss(d_pred_fake, False, dis_update=True)
        d_loss_real = self.gan_loss(d_pred_real, True, dis_update=True)
        d_loss = (d_loss_fake + d_loss_real) / 2.0

        self.manual_backward(d_loss)
        d_optimizer.step()
        d_optimizer.zero_grad()
        self.untoggle_optimizer(d_optimizer)

        self.log('training_losses/d_fake', d_loss_fake)
        self.log('training_losses/d_real', d_loss_real)
        self.log('training_losses/d_all', d_loss)

        if debug_inputs:
            assert 0
