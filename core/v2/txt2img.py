import os
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from ddpm import LatentDiffusion

from ddim import DDIMSampler
from plms import PLMSSampler
from dpm_solver import DPMSolverSampler

torch.set_grad_enabled(False)

def main():
    seed_everything(42)

    pl_sd = torch.load("core/v2/stable-diffusion_v2-1_768-ema-pruned.ckpt", map_location="cpu")
    sd = pl_sd["state_dict"]
    model = LatentDiffusion(
        parameterization = "v",
        linear_start = 0.00085,
        linear_end = 0.0120,
        num_timesteps_cond = 1,
        log_every_t = 200,
        timesteps = 1000,
        first_stage_key = "jpg",
        cond_stage_key = "txt",
        image_size = 64,
        channels = 4,
        cond_stage_trainable = False,
        conditioning_key = "crossattn",
        monitor = "val/loss_simple_ema",
        scale_factor = 0.18215,
        use_ema = False # we set this to false because this is an inference only config
    )
    
    model.load_state_dict(sd, strict=False)

    model.cuda()
    model.eval()

    # sampler = PLMSSampler(model, device=device)
    # sampler = DPMSolverSampler(model, device=device)
    sampler = DDIMSampler(model, device=torch.device("cuda"))

    outdir = "outputs/txt2img-samples"
    os.makedirs(outdir, exist_ok=True)

    batch_size = 1

    sample_count = 0
    base_count = len(os.listdir(outdir))

    start_code = None
    scale = 9.0
    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        uc = None
        if scale != 1.0:
            uc = model.get_learned_conditioning(batch_size * [""])
        c = model.get_learned_conditioning(["a professional photograph of an astronaut riding a triceratops"])

        shape = [4, 768 // 8, 768 // 8]
        samples, _ = sampler.sample(
            S=50,
            conditioning=c,
            batch_size=1,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            eta=0.0,
            x_T=start_code
        )

        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

        for x_sample in x_samples:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))
            img.save(os.path.join(outdir, f"{base_count:05}.png"))
            base_count += 1
            sample_count += 1


if __name__ == "__main__":
    main()
