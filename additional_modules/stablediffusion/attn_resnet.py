# Modified from https://github.com/Stability-AI/stablediffusion/blob/main/ldm/modules/diffusionmodules/model.py
import torch.nn as nn

from additional_modules.stablediffusion.modules import ResnetBlock, make_attn, Downsample, Upsample


class AttnResNet(nn.Module):
    def __init__(
        self,
        in_res=256,
        in_channels=96,
        out_channels=96,
        ch=128,
        attn_resolutions=[64, 32],
        skip_resolutions=None,
        num_res_blocks=2,
        ch_mult=[1, 2, 4]
    ):
        super().__init__()
        attn_type = 'vanilla-xformers'

        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.skip_resolutions = skip_resolutions
        self.ch = ch

        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,)+tuple(ch_mult)
        curr_res = in_res

        # Dowm blocks
        self.down = nn.ModuleList()
        for res_idx in range(self.num_resolutions):
            block = nn.ModuleList() 
            attn = nn.ModuleList()

            block_in = ch * in_ch_mult[res_idx]
            block_out = ch * ch_mult[res_idx]

            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=0,
                                         dropout=0.0))

                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if res_idx != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, True)
                curr_res = curr_res // 2

            self.down.append(down)

        # middle blocks
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=0,
                                       dropout=0.0)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=0,
                                       dropout=0.0)

        # Up blocks
        self.up = nn.ModuleList()
        for res_idx in reversed(range(0, self.num_resolutions, 1)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * in_ch_mult[res_idx]

            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=0,
                                         dropout=0.0))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))

            up = nn.Module()
            up.block = block
            up.attn = attn
            if res_idx != self.num_resolutions - 1:
                up.upsample = Upsample(block_in, True)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.normal_(self.conv_out.weight, mean=0.0, std=1e-5)

    def forward(self, planes):
        h = self.conv_in(planes)

        hs = [h]
        for res_idx in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[res_idx].block[i_block](h, None)
                if len(self.down[res_idx].attn) > 0:
                    h = self.down[res_idx].attn[i_block](h)

            if res_idx != self.num_resolutions-1:
                h = self.down[res_idx].downsample(h)
            hs.append(h)
        hs.pop()

        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)

        # upsampling
        for res_idx in reversed(range(0, self.num_resolutions, 1)):
            for i_block in range(self.num_res_blocks):
                h = self.up[res_idx].block[i_block](h, None)
                if len(self.up[res_idx].attn) > 0:
                    h = self.up[res_idx].attn[i_block](h)
            if res_idx != self.num_resolutions - 1:
                h = self.up[res_idx].upsample(h)

            if self.skip_resolutions is None or h.shape[-1] in self.skip_resolutions:
                h = h + hs.pop()
            else:
                hs.pop()

        out = self.conv_out(h)
        return out
