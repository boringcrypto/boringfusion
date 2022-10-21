import os
import numpy as np
from PIL import Image
from einops import rearrange
from transformers import logging
from core.model import DIFF_MODEL, DiffusionModel
from core.samplers import EulerASampler

logging.set_verbosity_error()

def save_samples(samples):
    base_count = len(os.listdir("outputs"))
    for sample in samples:
        sample = 255. * rearrange(sample.cpu().numpy(), 'c h w -> h w c')
        Image.fromarray(sample.astype(np.uint8)).save(os.path.join("outputs", f"{base_count:05}.png"))
        base_count += 1

def main():
    model = DiffusionModel(DIFF_MODEL.StableDiffusion1_5)

    seed = 42

    for i in range(10):
        samples = EulerASampler(model).sample(
            seed,
            512, 512, 1,
            "photo of a couch, photorealistic", 
            "", 7.5, 20
        )

        save_samples(samples)

        seed += 1

if __name__ == "__main__":
    main()
