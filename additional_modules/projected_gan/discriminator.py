import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.spectral_norm import SpectralNorm
from torchvision.transforms import Normalize, RandomCrop

import numpy as np
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torch_utils.ops import bias_act
from additional_modules.projected_gan.vit import make_vit_backbone, forward_vit
from additional_modules.projected_gan.diffaug import DiffAugment


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x) + x) / np.sqrt(2)


class FullyConnectedLayer(nn.Module):
    def __init__(
        self,
        in_features: int,              # Number of input features.
        out_features: int,             # Number of output features.
        bias: bool = True,             # Apply additive bias before the activation function?
        activation: str = 'linear',    # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier: float = 1.0,    # Learning rate multiplier.
        weight_init: float = 1.0,      # Initial standard deviation of the weight tensor.
        bias_init: float = 0.0,        # Initial value for the additive bias.
    ):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
        self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self) -> str:
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'


def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            padding_mode='circular',
        ),
        BatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-5):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()

        # Reshape batch into groups.
        G = np.ceil(x.size(0)/self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))

        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


class DiscHead(nn.Module):
    def __init__(self, channels: int, c_dim: int, cmap_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.cmap_dim = cmap_dim

        self.main = nn.Sequential(
            make_block(channels, kernel_size=1),
            ResidualBlock(make_block(channels, kernel_size=9))
        )

        if self.c_dim > 0:
            self.cmapper = FullyConnectedLayer(self.c_dim, cmap_dim)
            self.cls = SpectralConv1d(channels, cmap_dim, kernel_size=1, padding=0)
        else:
            self.cls = SpectralConv1d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.main(x)
        out = self.cls(h)

        if self.c_dim > 0:
            cmap = self.cmapper(c).unsqueeze(-1)
            out = (out * cmap).sum(1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return out


class DINO(torch.nn.Module):
    def __init__(self, hooks: list[int] = [2, 5, 8, 11], hook_patch: bool = True):
        super().__init__()
        self.n_hooks = len(hooks) + int(hook_patch)

        self.model = make_vit_backbone(
            timm.create_model('vit_small_patch16_224_dino', pretrained=True),
            patch_size=[16, 16], hooks=hooks, hook_patch=hook_patch,
        )
        self.model = self.model.eval().requires_grad_(False)

        self.img_resolution = self.model.model.patch_embed.img_size[0]
        self.embed_dim = self.model.model.embed_dim
        self.norm = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' input: x in [0, 1]; output: dict of activations '''
        x = F.interpolate(x, self.img_resolution, mode='area')
        x = self.norm(x)
        features = forward_vit(self.model, x)
        return features


class ProjectedDiscriminator(nn.Module):
    def __init__(self, c_dim: int, diffaug: bool = True, p_crop: float = 0.5):
        super().__init__()
        self.c_dim = c_dim
        self.diffaug = diffaug
        self.p_crop = p_crop

        self.dino = DINO()

        heads = []
        for i in range(self.dino.n_hooks):
            heads += [str(i), DiscHead(self.dino.embed_dim, c_dim)],
        self.heads = nn.ModuleDict(heads)

    def train(self, mode: bool = True):
        self.dino = self.dino.train(False)
        self.heads = self.heads.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Apply augmentation (x in [-1, 1]).
        if self.diffaug:
            x = DiffAugment(x, policy='color,translation,cutout')

        # Transform to [0, 1].
        x = x.add(1).div(2)

        # Take crops with probablity p_crop if the image is larger.
        if x.size(-1) > self.dino.img_resolution and np.random.random() < self.p_crop:
            x = RandomCrop(self.dino.img_resolution)(x)

        # Forward pass through DINO ViT.
        features = self.dino(x)

        # Apply discriminator heads.
        logits = []
        for k, head in self.heads.items():
            logits.append(head(features[k], c).view(x.size(0), -1))
        logits = torch.cat(logits, dim=1)

        return logits
