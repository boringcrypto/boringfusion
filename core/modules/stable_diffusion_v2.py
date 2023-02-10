import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from contextlib import contextmanager
from functools import partial

from .util import exists, default, make_beta_schedule, extract_into_tensor
from .stable_unet_v2 import UNetModel


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2

class LatentDiffusion(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 parameterization="eps",  # all assuming fixed variance schedules
                 ):
        super().__init__()
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.model = DiffusionWrapper()

        betas = make_beta_schedule()
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.num_timesteps = 1000
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if not isinstance(cond, list):
            cond = [cond]

        x_recon = self.model(x_noisy, t, cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon


class DiffusionWrapper(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.diffusion_model = UNetModel(
            use_fp16 = True,
            in_channels = 4,
            out_channels = 4,
            model_channels = 320,
            attention_resolutions = [ 4, 2, 1 ],
            num_res_blocks = 2,
            channel_mult = [ 1, 2, 4, 4 ],
            num_head_channels = 64,
            use_spatial_transformer = True,
            use_linear_in_transformer = True,
            transformer_depth = 1,
            context_dim = 1024,
            legacy = False
        )

    def forward(self, x, t, c_crossattn: list = None):
        cc = torch.cat(c_crossattn, 1)
        if hasattr(self, "scripted_diffusion_model"):
            # TorchScript changes names of the arguments
            # with argument cc defined as context=cc scripted model will produce
            # an error: RuntimeError: forward() is missing value for argument 'argument_3'.
            out = self.scripted_diffusion_model(x, t, cc)
        else:
            out = self.diffusion_model(x, t, context=cc)

        return out


