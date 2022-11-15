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

        self.set(layers)
        
        print("Setting eval")
        self.eval()

        # self.script = None

    def set(self, layers):
        print("Loading State")
        if layers is not None:
            self.diffusion_model.load_state_dict(layers)

    def forward(self, input_latent, noisiness, c_concat: list = None, c_crossattn: list = None):
        input_latent = input_latent.to(self.device, memory_format=torch.channels_last)
        noisiness = noisiness.to(self.device)
        cc = torch.cat(c_crossattn, 1).to(self.device)

        # if self.script is None:
        #     self.script = torch.jit.trace(self.diffusion_model.eval(), (x, t, cc))
        #     self.script.eval()
        #     self.script = torch.jit.optimize_for_inference(self.script)

        #     self.script.save("unet.ts")

            # self.script = torch.jit.load("unet.ts")
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

