import torch
from .autoencoder import AutoencoderKL

import torch
import numpy as np
import pytorch_lightning as pl
from functools import partial

from models.util import disabled_train, make_beta_schedule
from models.openaimodel import UNetModel


class StableDiffusionUNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.diffusion_model = UNetModel()

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        cc = torch.cat(c_crossattn, 1)
        out = self.diffusion_model(x, t, context=cc)

        return out


class LatentDiffusion(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.first_stage_model = AutoencoderKL().eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False        
        self.model = StableDiffusionUNet().eval()
        for param in self.model.parameters():
            param.requires_grad = False        

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
        self.eval()
        for param in self.parameters():
            param.requires_grad = False        

    def init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")

    @torch.no_grad()
    def decode_first_stage(self, z):
        print("Decoding first stage")
        z = 1. / 0.18215 * z # scale factor
        return self.first_stage_model.decode(z)

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if not isinstance(cond, list):
            cond = [cond]
        key = 'c_crossattn'
        cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

