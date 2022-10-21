import random
import torch
from models.ddim import DDIMSampler as rootDDIMSampler
from models.plms import PLMSSampler as rootPLMSSampler
import k_diffusion
from pytorch_lightning import seed_everything
from transformers import logging
from einops import repeat
import numpy as np
from PIL import Image
logging.set_verbosity_error()

class CFGDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

RANDOM_SEED = -1

def load_img(path):
    image = Image.open(path).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

class Sampler():
    def __init__(self, model) -> None:
        self.model = model

    def _sample(self, batch_size, c, uc, cfg):
        pass

    def _sample_image(self, batch_size, c, uc, cfg, steps, init_latent, strength, shape):
        pass

    def set_seed(self, seed):
        seed_everything(seed if seed != RANDOM_SEED else random.randint(0, 1000000000))

    @torch.no_grad()
    @torch.autocast("cuda")
    def sample(self, seed: int, width: int, height: int, batch_size: int, prompt: str or torch.Tensor, exclude: str or torch.Tensor, cfg: float, steps: int):
        """Create images from prompt

        Args:
            seed (integer): A seed or RANDOM_SEED to use a random seed.
            width (integer): Width of the images
            height (integer): Height of the images
            batch_size (integer): Number of images generated simultaneously on the GPU. Higher nunmber takes more VRAM.
            prompt (str|Tensor): Prompt for the image as string or latent space Tensor
            exclude (str|Tensor): Negative prompt for the image as string or latent space Tensor
            cfg (float): CFG, scale or strength of the prompt (2 is low, 7.5 is normal, 15+ is high)
            steps (integer): Number of denoising steps

        Returns:
            _type_: _description_
        """
        self.set_seed(seed)
        shape = [4, height // 8, width // 8]

        c = self.model.get_learned_conditioning(batch_size * [prompt]) if isinstance(prompt, str) else prompt
        uc = self.model.get_learned_conditioning(batch_size * [exclude]) if isinstance(exclude, str) else exclude

        samples = self._sample(batch_size, c, uc, cfg, steps, shape)
        samples = self.model.decode_first_stage(samples)
        return torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)

    @torch.no_grad()
    @torch.autocast("cuda")
    def sample_img(self, seed: int, width: int, height: int, batch_size: int, init_image, prompt: str or torch.Tensor, exclude: str or torch.Tensor, cfg: float, strength: float, steps: int):
        """Create images from images

        Args:
            seed (integer): A seed or RANDOM_SEED to use a random seed.
            width (integer): Width of the images
            height (integer): Height of the images
            batch_size (integer): Number of images generated simultaneously on the GPU. Higher nunmber takes more VRAM.
            prompt (str|Tensor): Prompt for the image as string or latent space Tensor
            exclude (str|Tensor): Negative prompt for the image as string or latent space Tensor
            cfg (float): CFG, scale or strength of the prompt (2 is low, 7.5 is normal, 15+ is high)
            steps (integer): Number of denoising steps

        Returns:
            _type_: _description_
        """
        self.set_seed(seed)
        shape = [4, height // 8, width // 8]

        init_image = load_img(init_image).to("cuda")
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))  # move to latent space

        c = self.model.get_learned_conditioning(batch_size * [prompt]) if isinstance(prompt, str) else prompt
        uc = self.model.get_learned_conditioning(batch_size * [exclude]) if isinstance(exclude, str) else exclude

        samples = self._sample_image(batch_size, c, uc, cfg, steps, init_latent, strength, shape)
        samples = self.model.decode_first_stage(samples)
        return torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)


class BaseSampler(Sampler):
    def __init__(self, model, sampler_class) -> None:
        super().__init__(model)
        self.sampler = sampler_class(model)

    def _sample(self, batch_size, c, uc, cfg, steps, shape):
        samples, _ = self.sampler.sample(S=steps,
            conditioning=c,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=cfg,
            unconditional_conditioning=uc,
            eta=0.0,
            x_T=None)
        return samples

    def _sample_image(self, batch_size, c, uc, cfg, steps, init_latent, strength, shape):
        self.sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0, verbose=False)
        t_enc = int(strength * steps)

        # encode (scaled latent)
        z_enc = self.sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to("cuda"))
        # decode it
        samples = self.sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=cfg, unconditional_conditioning=uc,)  
        return samples     

class KSampler(Sampler):
    def __init__(self, model) -> None:
        super().__init__(model)
        self.model_wrap = k_diffusion.external.CompVisDenoiser(model)
        self.sigma_min, self.sigma_max = self.model_wrap.sigmas[0].item(), self.model_wrap.sigmas[-1].item()

    def _inner(self):
        pass

    def _sample(self, batch_size, c, uc, cfg, steps, shape):
        sigmas = self.model_wrap.get_sigmas(steps)        
        x = torch.randn([batch_size, *shape], device="cuda") * sigmas[0]
        model_wrap_cfg = CFGDenoiser(self.model_wrap)
        extra_args = {'cond': c, 'uncond': uc, 'cond_scale': cfg}
        samples = self._inner(model_wrap_cfg, x, sigmas, steps, extra_args=extra_args)
        return samples

    def _sample_image(self, batch_size, c, uc, cfg, steps, init_latent, strength, shape):
        t_enc = int(strength * steps)   
        sigmas = self.model_wrap.get_sigmas(steps)
        sigma_sched = sigmas[steps - t_enc - 1:]
        noise = torch.randn([batch_size, *shape], device="cuda") * sigmas[0]
        x = init_latent + (noise * sigma_sched[0])
        model_wrap_cfg = CFGDenoiser(self.model_wrap)
        extra_args = {'cond': c, 'uncond': uc, 'cond_scale': cfg}

        self.sigma_min = sigma_sched[-2]
        self.sigma_max = sigma_sched[0]

        samples = self._inner(model_wrap_cfg, x, sigmas, steps, extra_args=extra_args)
        return samples

class DDIMSampler(BaseSampler):
    def __init__(self, model) -> None:
        super().__init__(model, rootDDIMSampler)

class PLMSSampler(BaseSampler):
    def __init__(self, model) -> None:
        super().__init__(model, rootPLMSSampler)

class EulerSampler(KSampler):
    def __init__(self, model) -> None:
        super().__init__(model)

    def _inner(self, model_wrap_cfg, x, sigmas, steps, extra_args):
        return k_diffusion.sampling.sample_euler(model_wrap_cfg, x, sigmas, extra_args)

class EulerASampler(KSampler):
    def __init__(self, model) -> None:
        super().__init__(model)

    def _inner(self, model_wrap_cfg, x, sigmas, steps, extra_args):
        return k_diffusion.sampling.sample_euler_ancestral(model_wrap_cfg, x, sigmas, extra_args)

class DPMAdaptiveSampler(KSampler):
    def __init__(self, model) -> None:
        super().__init__(model)

    def _inner(self, model_wrap_cfg, x, sigmas, steps, extra_args):
        return k_diffusion.sampling.sample_dpm_adaptive(model_wrap_cfg, x, self.sigma_min, self.sigma_max, extra_args)

class DPMFastSampler(KSampler):
    def __init__(self, model) -> None:
        super().__init__(model)

    def _inner(self, model_wrap_cfg, x, sigmas, steps, extra_args):
        return k_diffusion.sampling.sample_dpm_fast(model_wrap_cfg, x, self.sigma_min, self.sigma_max, steps, extra_args)

class DPM2Sampler(KSampler):
    def __init__(self, model) -> None:
        super().__init__(model)

    def _inner(self, model_wrap_cfg, x, sigmas, steps, extra_args):
        return k_diffusion.sampling.sample_dpm_2(model_wrap_cfg, x, sigmas, extra_args)

class DPM2ASampler(KSampler):
    def __init__(self, model) -> None:
        super().__init__(model)

    def _inner(self, model_wrap_cfg, x, sigmas, steps, extra_args):
        return k_diffusion.sampling.sample_dpm_2_ancestral(model_wrap_cfg, x, sigmas, extra_args)

class HeunSampler(KSampler):
    def __init__(self, model) -> None:
        super().__init__(model)

    def _inner(self, model_wrap_cfg, x, sigmas, steps, extra_args):
        return k_diffusion.sampling.sample_heun(model_wrap_cfg, x, sigmas, extra_args)

class LMSSampler(KSampler):
    def __init__(self, model) -> None:
        super().__init__(model)

    def _inner(self, model_wrap_cfg, x, sigmas, steps, extra_args):
        return k_diffusion.sampling.sample_lms(model_wrap_cfg, x, sigmas, extra_args)
