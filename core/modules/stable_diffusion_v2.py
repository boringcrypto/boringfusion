import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from contextlib import contextmanager
from functools import partial

from .util import exists, default, make_beta_schedule, extract_into_tensor
from .stable_unet_v2 import UNetModel
from enum import Enum
from .util import BoringModuleMixin

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class StableDiffusion(pl.LightningModule, BoringModuleMixin):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(
            self,
            version=1,
            layers=None,
            parameterization="eps",
            use_fp16=False,
            tiling=False,
            device="cuda"
        ):
        super().__init__()

        if version==1:
            self.parameterization = "eps"
            context_dim = 768
            use_linear_in_transformer = False
            num_head_channels = -1
            num_heads = 8
        else:
            self.parameterization = parameterization
            context_dim = 1024
            use_linear_in_transformer = True
            num_head_channels = 64
            num_heads = -1

        self.diffusion_model = UNetModel(
            use_fp16 = use_fp16,
            padding_mode = "circular" if tiling else "zeros",
            in_channels = 4,
            out_channels = 4,
            model_channels = 320,
            attention_resolutions = [ 4, 2, 1 ],
            num_res_blocks = 2,
            channel_mult = [ 1, 2, 4, 4 ],
            num_head_channels = num_head_channels,
            num_heads = num_heads,
            use_spatial_transformer = True,
            use_linear_in_transformer = use_linear_in_transformer,
            transformer_depth = 1,
            context_dim = context_dim,
            legacy = False
        )

        # def check(layer):
        #     if type(layer) == torch.nn.Conv2d:
        #         print(layer.padding_mode)
        #     for child in layer.children():
        #         check(child)

        # check(self.diffusion_model)

        if isinstance(layers, Enum):
            from ..data.model_data import ModelData
            layers = ModelData.load(layers, device)
            if use_fp16:
                layers.half_()

        self.set(layers)
        del layers

    def set(self, layers):
        print("Loading State")
        if layers is not None:
            self.diffusion_model.load_state_dict(layers)

    def forward(self, x, t, c_crossattn: list = None):
        cc = torch.cat(c_crossattn, 1)
        out = self.diffusion_model(x, t, context=cc)

        return out

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if not isinstance(cond, list):
            cond = [cond]

        x_recon = self(x_noisy, t, cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon


class StableDiffusionUpscaler(pl.LightningModule, BoringModuleMixin):
    def __init__(
            self,
            layers=None,
            parameterization="v",
            use_fp16=False,
            tiling=False,
            device="cuda"
        ):
        super().__init__()

        self.parameterization = parameterization

        self.diffusion_model = UNetModel(
            use_fp16 = use_fp16,
            padding_mode = "circular" if tiling else "zeros",
            in_channels = 7,
            out_channels = 4,
            model_channels = 256,
            attention_resolutions = [ 2, 4, 8 ],
            num_res_blocks = 2,
            channel_mult = [ 1, 2, 2, 4 ],
            num_head_channels = -1,
            num_heads = 8,
            use_spatial_transformer = True,
            use_linear_in_transformer = True,
            transformer_depth = 1,
            context_dim = 1024,
            legacy = False,
            disable_self_attentions = [True, True, True, False],
            disable_middle_self_attn = False,
            num_classes = 1000,
            image_size = 128
        )

        if isinstance(layers, Enum):
            from ..data.model_data import ModelData
            layers = ModelData.load(layers, device)
            if use_fp16:
                layers.half_()

        self.set(layers)
        del layers

    def set(self, layers):
        print("Loading State")
        if layers is not None:
            self.diffusion_model.load_state_dict(layers)

    def forward(self, x, t, c_crossattn: list = None):
        cc = torch.cat(c_crossattn, 1)
        out = self.diffusion_model(x, t, context=cc)

        return out

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if not isinstance(cond, list):
            cond = [cond]

        x_recon = self(x_noisy, t, cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon



    #     self.instantiate_low_stage(low_scale_config)
    #     self.low_scale_key = low_scale_key
    #     self.noise_level_key = noise_level_key

    # def instantiate_low_stage(self, config):
    #     model = instantiate_from_config(config)
    #     self.low_scale_model = model.eval()
    #     self.low_scale_model.train = disabled_train
    #     for param in self.low_scale_model.parameters():
    #         param.requires_grad = False

