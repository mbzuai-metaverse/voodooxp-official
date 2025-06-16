# https://youtu.be/n68BF6IlGhk?si=0YdNah_x4ThoxSUX
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Tuple

from additional_modules.segformer.backbone import Block, OverlapPatchEmbed
from additional_modules.stablediffusion.attention import BasicTransformerBlock
from models.lp3d_model import Lp3DELow, Lp3DEHigh, Lp3DLightningModel, Lp3DwRenderer
from utils.registry import MODEL_REGISTRY


class NeutralizerELow(Lp3DELow):
    def __init__(self, img_size: int = 512, img_channels: int = 3):
        super().__init__(img_size, img_channels)
        self.neutral_attn1 = BasicTransformerBlock(dim=1024, n_heads=4, d_head=512, context_dim=None)
        self.neutral_attn2 = BasicTransformerBlock(dim=1024, n_heads=4, d_head=512, context_dim=None)
        self.neutral_attn3 = BasicTransformerBlock(dim=1024, n_heads=4, d_head=512, context_dim=None)
        self.neutral_attn4 = BasicTransformerBlock(dim=1024, n_heads=4, d_head=512, context_dim=None)
        self.neutral_attn5 = BasicTransformerBlock(dim=1024, n_heads=4, d_head=512, context_dim=None)

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


class NeutralizerLp3DEncoder(nn.Module):
    def __init__(self, img_size: int = 512, img_channels: int = 3, triplane_nd: int = 32):
        super().__init__()
        self.img_size = img_size

        self.elo = NeutralizerELow(img_size, img_channels)
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


class NeutralizerwRenderer(Lp3DwRenderer):
    def __init__(self, neural_rendering_resolution, triplane_nd, **rendering_kwargs):
        super().__init__(neural_rendering_resolution, triplane_nd, **rendering_kwargs)

        del self.triplane_encoder
        del self.superresolution
        torch.cuda.empty_cache()
        self.triplane_encoder = NeutralizerLp3DEncoder(triplane_nd=triplane_nd)

        for k, v in self.named_parameters():
            if 'neutral' not in k:
                v.requires_grad_(False)

    def forward(self, source_image: torch.Tensor) -> torch.Tensor:
        """ 
        Neutralize the source image. Returns the neutralized triplanes.

        Parameters:
            - source_image (Tensor): The source image.
        Returns:
            - planes (Tensor): Neutralized triplanes
        """

        planes = self.canonicalize(source_image)
        return planes


@MODEL_REGISTRY.register()
class NeutralizerLightningModel(Lp3DLightningModel):
    def __init__(
        self,
        neural_rendering_resolution: int,
        subsampling_ratio: float,
        triplane_nd: int,
        triplane_h: int,
        triplane_w: int,
        pretrained_path: str,
        eg3d_network_pkl: str,
        rendering_kwargs,
        superresolution_kwargs,
        loss_kwargs,
        training_kwargs,
        val_kwargs
    ):
        super().__init__(
            neural_rendering_resolution,
            subsampling_ratio,
            triplane_nd,
            triplane_h,
            triplane_w,
            pretrained_path,
            eg3d_network_pkl,
            rendering_kwargs,
            superresolution_kwargs,
            loss_kwargs,
            training_kwargs,
            val_kwargs
        )

        self.register_buffer('lookat_position', torch.tensor([0, 0, 0]))
        self.automatic_optimization = True

    def _setup_modules(self):
        super()._setup_modules()

        self.neutralizer = NeutralizerwRenderer(triplane_nd=self.triplane_nd)
        self.lp3d.requires_grad_(False)

    def apply_model(
        self,
        xs_data: Dict[str, torch.Tensor],
        gt_data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the ground-truth triplanes and neutralized xs triplanes.

        Parameters:
            - xs_data: The source's data. Must have 'image' key in it.
            - gt_data: The ground-truth's data. Must have 'image' key in it.
        Returns:
            - source_planes: Source's triplanes.
            - gt_planes: Ground-truth's triplanes.
        """
        source_planes = self.neutralizer(xs_data['image'])
        with torch.no_grad():
            gt_planes = self.lp3d(gt_data['image'])

        return source_planes, gt_planes

    def training_step(self, batch, batch_idx):
        xs_data = batch['xs_data']
        xd_data = batch['xd_data']

        # loss weights
        triplane_weight = self.loss_opt['triplane']['weight']

        source_planes, gt_planes = self.apply_model(xs_data, xd_data)

        batch_size = source_planes.shape[0]
        device = source_planes.device
        lookat_position = self.lookat_position.unsqueeze(0).repeat(batch_size, 1)
        cam2world = self.pose_sampler.sample(
            0.71, 1.11, 2.7,
            lookat_position,
            2.42, 2.02, 0.1,
            batch_size=batch_size, device=device
        )
        intrinsics = self.intrinsics_sampler.sample(
            18.837, 0.5,
            1.5, 0.02,
            batch_size=batch_size, device=device
        )

        source_rendered = self.neutralizer.render(source_planes, cam2world, intrinsics, upsample=False)
        gt_rendered = self.neutralizer.render(gt_planes, cam2world, intrinsics, upsample=False)

        g_loss = self._calculate_image_loss(source_rendered, gt_rendered, 'image_loss')

        # Triplane loss
        if triplane_weight > 0:
            triplane_loss = F.l1_loss(source_planes, gt_planes, reduce=True)
            g_loss += triplane_weight * triplane_loss
            self.log(f'training_losses/triplane', triplane_loss)

        return g_loss

    def configure_optimizers(self):
        g_params = list(self.neutralizer.parameters())
        g_params = filter(lambda x: x.requires_grad, g_params)
        g_optimizer = torch.optim.AdamW(
            g_params,
            lr=self.training_opt['learning_rate'],
            betas=self.training_opt['betas'],
            eps=self.training_opt['eps'],
            weight_decay=self.training_opt['weight_decay']
        )

        return g_optimizer
