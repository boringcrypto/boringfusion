import random
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import k_diffusion
from pytorch_lightning import seed_everything
from transformers import logging
from einops import repeat
import numpy as np
from PIL import Image
from .modules.ddim import DDIMSampler as rootDDIMSampler
from .modules.plms import PLMSSampler as rootPLMSSampler
from .modules.clip import PromptBuilder
from .modules.util import BoringModule, should_run_on_gpu
logging.set_verbosity_error()

class CFGDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])

        # Short explanation of how the scale/guidance works
        # https://www.youtube.com/watch?v=_7rMfsA24Ls&t=1943s
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


class CFGDenoiserSlew(nn.Module):
    '''
    Clamps the maximum change each step can have.
    "limit" is the clamp bounds. 0.4-0.8 seem good, 1.6 and 3.2 have very little difference and might represent the upper bound of values.
    "blur" is the radius of a gaussian blur used to split the limited output with the original output in an attempt to preserve detail and color.
    "last_step_is_blur" if true will compare the model output to the blur-split output rather than just the limited output, can look nicer.
    '''
    def __init__(self, model, limit = 0.6, blur = 5, last_step_is_blur = True):
        super().__init__()
        self.inner_model = model
        self.last_sigma = 0.0 # For keeping track of when the sampling cycle restarts for a new image
        self.last_step = None # For keeping the last step for measuring change between steps
        self.limit = limit # The clamp bounds
        self.blur = blur # Radius of the blur for freq splitting and merging limited and non-limited outputs
        self.last_step_is_blur = last_step_is_blur # Compare outputs to the freq split output instead of the plain limited output

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])

        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)

        # Short explanation of how the scale/guidance works
        # https://www.youtube.com/watch?v=_7rMfsA24Ls&t=1943s
        result_clean = uncond + (cond - uncond) * cond_scale

        if sigma > self.last_sigma:
            self.last_step = None
        self.last_sigma = sigma
        if self.last_step != None:
            diff = result_clean - self.last_step
            result = diff.clamp(-1 * self.limit, self.limit) + self.last_step
            if self.last_step_is_blur == False:
                self.last_step = result # Pre-blur
            if self.blur > 1:
                result = TF.gaussian_blur(result, self.blur)
                result_clean_hi = result_clean - TF.gaussian_blur(result_clean, self.blur)
                result = result + result_clean_hi
                if self.last_step_is_blur == True:
                    self.last_step = result # Post-blur
                del result_clean_hi
            del diff, x_in, sigma_in, cond_in, uncond, cond, result_clean
        else:
            result = result_clean
            self.last_step = result
        return result

RANDOM_SEED = -1

def load_img(path):
    image = Image.open(path).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

class Sampler(BoringModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def _sample(self, batch_size, c, uc, cfg):
        pass

    def _sample_image(self, batch_size, c, uc, cfg, steps, init_latent, strength, shape):
        pass

    def set_seed(self, seed):
        seed_everything(seed if seed != RANDOM_SEED else random.randint(0, 1000000000))

    @torch.no_grad()
    def sample(self, seed: int, width: int, height: int, batch_size: int, prompt: torch.Tensor or PromptBuilder, negative_prompt: torch.Tensor, cfg: float, steps: int):
        """Create images from prompt

        Args:
            seed (integer): A seed or RANDOM_SEED to use a random seed.
            width (integer): Width of the images
            height (integer): Height of the images
            batch_size (integer): Number of images generated simultaneously on the GPU. Higher nunmber takes more VRAM.
            prompt (Tensor|PromptBuilder): Prompt for the image as string or latent space Tensor
            exclude (str|Tensor): Negative prompt for the image as string or latent space Tensor
            cfg (float): CFG, scale or strength of the prompt (2 is low, 7.5 is normal, 15+ is high)
            steps (integer): Number of denoising steps

        Returns:
            _type_: _description_
        """
        self.set_seed(seed)
        shape = [4, height // 8, width // 8]

        if isinstance(prompt, PromptBuilder):
            prompt = prompt.embedding

        prompt = prompt.to(self.device)
        negative_prompt = negative_prompt.to(self.device)

        samples = self._sample(batch_size, prompt, negative_prompt, cfg, steps, shape)
        return samples


class BaseSampler(Sampler):
    def __init__(self, model, sampler_class) -> None:
        super().__init__(model)
        self.sampler = sampler_class(model)

    def _sample(self, batch_size, prompt, negative_prompt, cfg, steps, shape):
        samples = self.sampler.sample(S=steps,
            conditioning=prompt,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=cfg,
            unconditional_conditioning=negative_prompt,
            eta=0.0,
            x_T=None)
        return samples


class KSampler(Sampler):
    def __init__(self, model) -> None:
        super().__init__(model)
        self.model_wrap = k_diffusion.external.CompVisDenoiser(model)
        self.sigma_min, self.sigma_max = self.model_wrap.sigmas[0].item(), self.model_wrap.sigmas[-1].item()

    def _inner(self):
        pass

    def _sample(self, batch_size, prompt, negative_prompt, cfg, steps, shape):
        sigmas = self.model_wrap.get_sigmas(steps).to(self.device)   
        x = torch.randn([batch_size, *shape], device=self.device) * sigmas[0]
        model_wrap_cfg = CFGDenoiser(self.model_wrap)
        extra_args = {'cond': prompt, 'uncond': negative_prompt, 'cond_scale': cfg}
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
