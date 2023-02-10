import os
import torch
import numpy as np
import PIL
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from core.modules.stable_diffusion_v2 import LatentDiffusion

from core.modules.ddim2 import DDIMSampler
from core.modules.util import repeat
from core.modules.vae_decoder import VAEDecoder
from core.modules.vae_encoder import VAEEncoder
from core.modules.clip import OpenCLIPEmbedder
import models

torch.set_grad_enabled(False)

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
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
        image_size = 64,
        channels = 4,
        monitor = "val/loss_simple_ema",
        scale_factor = 0.18215,
    )
    
    model.load_state_dict(sd, strict=False)

    model.cuda()
    model.eval()

    clip = OpenCLIPEmbedder().cuda()

    sampler = DDIMSampler(model, device=torch.device("cuda"))

    outdir = "outputs/txt2img-samples"
    os.makedirs(outdir, exist_ok=True)

    batch_size = 1
    base_count = len(os.listdir(outdir))

    steps = 50
    scale = 9.0

    # sampling
    shape = [4, 768 // 8, 768 // 8]
    C, H, W = shape
    size = (batch_size, C, H, W)
    noise = torch.randn(size, device="cuda")

    with torch.no_grad(), autocast("cuda"):
        sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0.0, verbose=False)

        # decode it
        samples = sampler.decode(
            noise,
            clip("french park bench, photorealistic, 4k"),
            steps,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=clip("")
        )

    x_samples = VAEDecoder(models.decoder.VAEDec1_5mse_fp32)(samples)

    for x_sample in x_samples:
        x_sample.save("start.png")
        base_count += 1


    init_img = "start.png"
    init_image = load_img(init_img).to("cuda")
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)

    init_latent = VAEEncoder(models.encoder.VAEEnc1_4_fp32)(init_image)

    uc = clip("")
    c = clip("overgrown and worn french park bench")

    noise = torch.randn(size, device="cuda")

    for i in range(35, 36):
        actual_steps = i
        with torch.no_grad(), autocast("cuda"):
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

        x_samples = VAEDecoder(models.decoder.VAEDec1_5mse_fp32)(samples)

        for x_sample in x_samples:
            x_sample.save(os.path.join(outdir, f"{base_count:05}.png"))
            base_count += 1



if __name__ == "__main__":
    main()
