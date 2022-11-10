import torch
import numpy as np
import pytorch_lightning as pl
from functools import partial

from .util import BoringModuleMixin, make_beta_schedule
from .stable_unet import UNetModel


class StableDiffusion(pl.LightningModule, BoringModuleMixin):
    def __init__(self, layers=None, use_fp16=False, device="cuda"):
        super().__init__()
        self.diffusion_model = UNetModel(use_fp16=use_fp16, device=device)

        print("Making schedule")
        betas = make_beta_schedule("linear", 1000, linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        to_torch = partial(torch.tensor, dtype=self.dtype, device=device)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.set(layers)
        
        print("Setting eval")
        self.eval()

        # self.script = None

    def set(self, layers):
        print("Loading State")
        if layers is not None:
            self.diffusion_model.load_state_dict(layers)

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        x = x.to(self.device, memory_format=torch.channels_last)
        t = t.to(self.device)
        cc = torch.cat(c_crossattn, 1).to(self.device)

        # if self.script is None:
        #     self.script = torch.jit.trace(self.diffusion_model.eval(), (x, t, cc))
        #     self.script.eval()
        #     self.script = torch.jit.optimize_for_inference(self.script)

        #     self.script.save("unet.ts")

            # self.script = torch.jit.load("unet.ts")
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

    @property
    def dtype(self):
        self.diffusion_model.dtype

