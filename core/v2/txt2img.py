import os
import torch
import numpy as np
import PIL
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from ddpm import LatentDiffusion

from ddim import DDIMSampler
from dpm_solver import DPMSolverSampler
from util import repeat

torch.set_grad_enabled(False)

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.

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

    # sampler = DPMSolverSampler(model, device=torch.device("cuda"))
    sampler = DDIMSampler(model, device=torch.device("cuda"))

    outdir = "outputs/txt2img-samples"
    os.makedirs(outdir, exist_ok=True)

    batch_size = 1

    sample_count = 0
    base_count = len(os.listdir(outdir))

    steps = 50
    strength = 0
    actual_steps = int(strength * steps)
    print(f"target t_enc is {actual_steps} steps")
    scale = 9.0

    # sampling
    shape = [4, 768 // 8, 768 // 8]
    C, H, W = shape
    size = (batch_size, C, H, W)

    init_img = "rick.jpeg"
    init_image = load_img(init_img).to("cuda")
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    noise = torch.randn(size, device="cuda")

    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        uc = None
        if scale != 1.0:
            uc = model.get_learned_conditioning(batch_size * [""])
        c = model.get_learned_conditioning(["michael jackson"])

        sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0.0, verbose=False)

        # encode (scaled latent)
        if actual_steps < steps:
            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([actual_steps] * batch_size).to("cuda"), noise=noise)
        else:
            z_enc = noise

        # decode it
        samples = sampler.decode(
            z_enc,
            c,
            actual_steps,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc
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
