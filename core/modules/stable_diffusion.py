import torch
import numpy as np
from enum import Enum
import pytorch_lightning as pl
from functools import partial

from .util import BoringModuleMixin
from .stable_unet import UNetModel


class StableDiffusion(pl.LightningModule, BoringModuleMixin):
    def __init__(self, layers=None, use_fp16=False, context_dim=768, use_linear_in_transformer=False, tiling=False, device="cuda"):
        super().__init__()
        self.diffusion_model = UNetModel(
            context_dim=context_dim,
            use_linear_in_transformer=use_linear_in_transformer,
            padding_mode="circular" if tiling else "zeros",
            use_fp16=use_fp16, 
            device=device
        )

        if isinstance(layers, Enum):
            from ..data.model_data import ModelData
            layers = ModelData.load(layers, device)
            if use_fp16:
                layers.half_()

        self.set(layers)
        del layers
        
        print("Setting eval")
        self.eval()
        self.to(device=device)

        # self.script = None

    def set(self, layers):
        print("Loading State")
        if layers is not None:
            self.diffusion_model.load_state_dict(layers)

    def forward(self, input_latent, noisiness, c_concat: list = None, c_crossattn: list = None):
        input_latent = input_latent.to(self.device, memory_format=torch.channels_last)
        noisiness = noisiness.to(self.device)
        cc = torch.cat(c_crossattn, 1).to(self.device)
        output_latent = self.diffusion_model(input_latent, noisiness, context=cc)

        return output_latent

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if not isinstance(cond, list):
            cond = [cond]
        key = 'c_crossattn'
        cond = {key: cond}

        x_recon = self(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    @property
    def dtype(self):
        self.diffusion_model.dtype

