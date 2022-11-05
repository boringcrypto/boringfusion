import torch
from .latent_decoder import LatentDecoder

import torch
import numpy as np
import pytorch_lightning as pl
from functools import partial

from .util import BoringModuleMixin, disabled_train, make_beta_schedule, should_run_on_gpu
from .stable_unet import UNetModel
from einops import rearrange
from PIL import Image


class StableDiffusion(pl.LightningModule, BoringModuleMixin):
    def __init__(self):
        super().__init__()
        self.diffusion_model = UNetModel()

        betas = make_beta_schedule("linear", 1000, linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        x = x.to(self.device)
        t = t.to(self.device)
        cc = torch.cat(c_crossattn, 1).to(self.device)
        
        out = self.diffusion_model(x, t, context=cc)

        return out

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



