import torch
import torch.nn as nn
from PIL import Image

from .util import BoringModule, should_run_on_gpu
from enum import Enum
from ..data.model_data import ModelData

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
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


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # z to block_in
        self.conv_in = torch.nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(512, 512)
        self.mid.attn_1 = AttnBlock(512)
        self.mid.block_2 = ResnetBlock(512, 512)

        # upsampling
        block_in = 512
        curr_res = 32
        self.up = nn.ModuleList()
        for i_level in reversed(range(4)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = 128*[1, 2, 4, 4][i_level]
            for i in range(3):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(4)):
            for i_block in range(3):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # torchvision.ToPILImage only works on a single image, so here we prepare all images
        # on the GPU all at once.

        # Normalize from float between -1.0 and 1.0 to floats between 0.0 and 255.0
        h = h.add_(1.0).mul_(127.5).clamp_(min=0.0, max=255.0).type(torch.uint8)
        # swap from the dimensions for PIL
        h = h.permute(0, 2, 3, 1)
        return h

class VAEDecoder(BoringModule):
    def __init__(self, layers=None):
        # TODO: Pass in device and create directly on that device
        super().__init__()

        if isinstance(layers, Enum):
            layers = ModelData.load(layers)

        self.decoder = Decoder()
        self.post_quant_conv = torch.nn.Conv2d(4, 4, 1)
        self.set(layers)
        self.eval()
        # self.script = None

    def set(self, layers):
        if layers is not None:
            self.load_state_dict(layers)

    @should_run_on_gpu
    @torch.no_grad()
    def forward(self, z):
        # Make sure the dtype matches
        z = z.type(self.decoder.conv_in.bias.dtype)
        # Scale by the fixed scale factor of the SD model
        z = 1. / 0.18215 * z # scale factor

        # Decode the image from latent space
        z = self.post_quant_conv(z)

        # if self.script is None:
        #     self.script = torch.jit.trace(self.decoder, z)
        #     self.script.save("vae_decoder.ts")
        #     torch.cuda.synchronize()

        # import time
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # for i in range(3):
        #     start.record()
        #     samples = self.decoder(z)
        #     end.record()
        #     torch.cuda.synchronize()
        #     print("Normal", start.elapsed_time(end))

        #     start.record()
        #     samples = self.script(z)
        #     end.record()
        #     torch.cuda.synchronize()
        #     print("Trace", start.elapsed_time(end))

        samples = self.decoder(z)

        # Move the samples to the cpu in into a numpy array
        samples = samples.cpu().numpy()
        
        # Convert all samples into PIL Images
        return [Image.fromarray(sample) for sample in samples]
