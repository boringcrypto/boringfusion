import os
import time
import numpy as np
import torch
from PIL import Image
from einops import rearrange
from core.model import DIFF_MODEL, DiffusionModel
from core.samplers import DDIMSampler, EulerASampler, PLMSSampler
from models.encoders import FrozenCLIPEmbedder

def save_samples(samples):
    base_count = len(os.listdir("outputs"))
    for sample in samples:
        sample = 255. * rearrange(sample.cpu().numpy(), 'c h w -> h w c')
        Image.fromarray(sample.astype(np.uint8)).save(os.path.join("outputs", f"{base_count:05}.png"))
        base_count += 1

def main():
    clip = FrozenCLIPEmbedder().eval().cuda()
    prompt = clip("photo of a couch, photorealistic")
    negative_prompt = clip("")
    torch.cuda.empty_cache()
    clip.cpu()
    time.sleep(5)

    model = DiffusionModel(DIFF_MODEL.StableDiffusion1_5)

    seed = 42

    for i in range(4):
        samples = EulerASampler(model).sample(
            seed,
            512, 512, 1,
            prompt, 
            negative_prompt, 7.5, 20
        )

        save_samples(samples)

        seed += 1

    for i in range(3):
        samples = DDIMSampler(model).sample(
            seed,
            512, 512, 1,
            prompt, 
            negative_prompt, 7.5, 20
        )

        save_samples(samples)

        seed += 1

    for i in range(3):
        samples = PLMSSampler(model).sample(
            seed,
            512, 512, 1,
            prompt, 
            negative_prompt, 7.5, 20
        )

        save_samples(samples)

        seed += 1

if __name__ == "__main__":
    main()
