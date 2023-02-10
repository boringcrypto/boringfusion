import torch
import torch.nn as nn
import numpy
from PIL import Image

from .util import BoringModule, should_run_on_gpu
from enum import Enum
from .attention import make_attn

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        batch, channels, height, width = q.shape
        q = q.reshape(batch, channels, height*width)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(batch, channels, height*width) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(channels)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(batch, channels, height*width)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(batch, channels, height, width)

        h_ = self.proj_out(h_)

        return x+h_

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        ch_mult=(1,2,4,4)

        # downsampling
        self.conv_in = torch.nn.Conv2d(3,
                                       128,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = 256
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(4):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = 128*in_ch_mult[i_level]
            block_out = 128*ch_mult[i_level]
            for i_block in range(2):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != 3:
                down.downsample = Downsample(block_in, True)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in)
        self.mid.attn_1 = make_attn(block_in, attn_type="vanilla")
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        8,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, torch_images):
        # downsampling
        hs = [self.conv_in(torch_images)]
        for i_level in range(4):
            for i_block in range(2):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != 3:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

class VAEEncoder(BoringModule):
    def __init__(self,
                 layers
                 ):
        super().__init__()

        if isinstance(layers, Enum):
            from ..data.model_data import ModelData
            layers = ModelData.load(layers)

        self.encoder = Encoder()
        self.quant_conv = torch.nn.Conv2d(8, 8, 1)
        self.set(layers)
        self.eval()

    def set(self, layers):
        if layers is not None:
            self.load_state_dict(layers)

    @should_run_on_gpu
    @torch.no_grad()
    def forward(self, image):
        image.to(self.device)
        # Make sure the dtype matches
        image.type(self.encoder.conv_in.bias.dtype)

        encoded = self.encoder(image)

        moments = self.quant_conv(encoded)
        posterior = DiagonalGaussianDistribution(moments).sample()
        latent = 0.18215 * posterior

        return latent


