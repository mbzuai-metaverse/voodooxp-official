import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2

from additional_modules.eg3d.camera_utils import IntrinsicsSampler, LookAtPoseSampler
from additional_modules.eg3d.networks import OSGDecoder
from additional_modules.segformer.backbone import Block, OverlapPatchEmbed
from additional_modules.stablediffusion.attention import BasicTransformerBlock
from models.lp3d_model import ELow, EHigh
from rendering.ray_sampler import RaySampler
from rendering.triplane_rendering.renderer import ImportanceRenderer
from utils.image_utils import tensor2img


PRETRAINED_ID_PATH = 'experiments/pretrained_models/id_extractor_iter34k.pth'


class ELowNeutralize(ELow):
    def __init__(self, img_size: int = 512, img_channels: int = 3):
        super().__init__(img_size, img_channels)
        self.neutral_attn1 = BasicTransformerBlock(dim=1024, n_heads=4, d_head=512, context_dim=None)
        self.neutral_attn2 = BasicTransformerBlock(dim=1024, n_heads=4, d_head=512, context_dim=None)
        self.neutral_attn3 = BasicTransformerBlock(dim=1024, n_heads=4, d_head=512, context_dim=None)
        self.neutral_attn4 = BasicTransformerBlock(dim=1024, n_heads=4, d_head=512, context_dim=None)
        self.neutral_attn5 = BasicTransformerBlock(dim=1024, n_heads=4, d_head=512, context_dim=None)

        self.clear_cache()

    def clear_cache(self):
        self.cache = None

    def forward(
        self,
        img: torch.Tensor,
    ):
        x = self._add_positional_encoding(img)
        x = self.deeplabv3_backbone(x)
        x, H, W = self.patch_embed(x)
        x = self.block1(x, H, W)

        x = self.neutral_attn1(x)

        x = self.block2(x, H, W)
        x = self.neutral_attn2(x)

        x = self.block3(x, H, W)
        x = self.neutral_attn3(x)

        x = self.block4(x, H, W)
        x = self.neutral_attn4(x)

        x = self.block5(x, H, W)
        x = self.neutral_attn5(x)

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


class Lp3DEncoderNeutralize(nn.Module):
    def __init__(self, img_size: int = 512, img_channels: int = 3, triplane_nd: int = 32):
        super().__init__()
        self.img_size = img_size

        self.elo = ELowNeutralize(img_size, img_channels)
        self.ehi = EHigh(img_size, img_channels)

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

        self.clear_cache()

    def clear_cache(self):
        self.f_hi = None
        self.elo.clear_cache()

    def forward(
        self,
        img: torch.Tensor,
    ):
        assert img.shape[-1] == self.img_size and img.shape[-2] == self.img_size

        f_lo = self.elo(img)
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


class NeutralizingLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.triplane_encoder = Lp3DEncoderNeutralize()

        self.renderer = ImportanceRenderer()
        self.decoder = OSGDecoder(
            32,
            {
                'decoder_lr_mul': 1,
                'decoder_output_dim': 32
            }
        )
        self.ray_sampler = RaySampler()

        state_dict = torch.load(PRETRAINED_ID_PATH, map_location='cpu')['state_dict']
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('triplane_encoder.') or k.startswith('decoder.') or k.startswith('renderer.'):
                clean_state_dict[k] = v

        self.load_state_dict(clean_state_dict, strict=True)

        self.register_buffer('lookat_position', torch.tensor([0, 0, 0]))
        self.requires_grad_(False)

        self.rendering_kwargs = {
          'decoder_lr_mul': 1.0,
          'depth_resolution': 96,
          'depth_resolution_importance': 96,
          'camera_radius': 2.7,
          'lookat_point': [0.0, 0.0, 0.2],
          'ray_start': 2.25,
          'ray_end': 3.3,
          'box_warp': 1,
          'disparity_space_sampling': False,
          'clamp_mode': 'softplus'
        }

    def render(self, planes, cam2world, intrinsics):
        ray_origins, ray_directions = self.ray_sampler(
            cam2world, intrinsics, 128
        )

        batch_size = cam2world.shape[0]

        feature_samples, _, _ = self.renderer(
            planes,
            self.decoder,
            ray_origins,
            ray_directions,
            self.rendering_kwargs
        )  # channels last

        H = W = 128
        feature_image = feature_samples.permute(0, 2, 1).reshape(batch_size, feature_samples.shape[-1], H, W).contiguous()
        rgb_image_raw = feature_image[:, :3]

        return rgb_image_raw

    def canonicalize(
        self, image: torch.Tensor,
    ):
        image = (image + 1) / 2.  # Legacy issue :(
        triplanes = self.triplane_encoder(image)
        B = triplanes.shape[0]
        triplanes = triplanes.view(B, 3, 32, triplanes.shape[-2], triplanes.shape[-1]).contiguous()

        return triplanes

    def forward(self, pred, gt):
        assert pred.shape[-1] == gt.shape[-1] and pred.shape[-1] == 512

        pred_triplane = self.canonicalize(pred)
        pred_triplane = F.interpolate(pred_triplane, (128, 128))

        gt_triplane = self.canonicalize(gt)
        gt_triplane = F.interpolate(gt_triplane, (128, 128))

        assert not gt_triplane.requires_grad and pred_triplane.requires_grad

        # if np.random.randint(5) == 0:
        #     batch_size = pred.shape[0]
        #     lookat_position = self.lookat_position.unsqueeze(0).repeat(batch_size, 1)
        #     with torch.no_grad():
        #         intrinsics = IntrinsicsSampler.sample(
        #             18.837, 0.5,
        #             1.5, 0.02,
        #             batch_size=batch_size, device=pred.device
        #         )
        #         cam2world = LookAtPoseSampler.sample(
        #             0.71, 1.11, 2.7,
        #             lookat_position,
        #             2.42, 2.02, 0.1,
        #             batch_size=batch_size, device=pred.device
        #         )
        #         pred_img = self.render(pred_triplane, cam2world, intrinsics)
        #         gt_img = self.render(gt_triplane, cam2world, intrinsics)

        #     pred_img = tensor2img(pred_img, min_max=(-1, 1))
        #     gt_img = tensor2img(gt_img, min_max=(-1, 1))
        #     pred_org = tensor2img(pred, min_max=(-1, 1))
        #     gt_org = tensor2img(gt, min_max=(-1, 1))
        #     cv2.imwrite('debug_neutral1.png', np.hstack((pred_img, gt_img)))
        #     cv2.imwrite('debug_neutral2.png', np.hstack((pred_org, gt_org)))

        return F.l1_loss(pred_triplane, gt_triplane)

