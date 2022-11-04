from base64 import decode
import gc
import os
import time
import numpy as np
import torch
from PIL import Image
from einops import rearrange
from core.model import DIFF_MODEL, DiffusionModel
from core.samplers import DDIMSampler, EulerASampler, PLMSSampler
from modules.clip import CLIPEmbedder
from modules.latent_decoder import LatentDecoder

def save_images(samples):
    base_count = len(os.listdir("outputs"))
    for sample in samples:
        sample.save(os.path.join("outputs", f"{base_count:05}.png"))
        base_count += 1

def main():
    clip = CLIPEmbedder()
    prompts = clip(["black woman", ""])

    model = DiffusionModel(DIFF_MODEL.StableDiffusion1_5)
    decoder = LatentDecoder().cuda()

    seed = 42

    for i in range(10):
        samples = EulerASampler(model).sample(
            seed + i,
            512, 768, 1,
            prompts[[0]], 
            prompts[[1]], 7.5, 20
        )

        images = decoder.latents_to_images(samples)

        save_images(images)

if __name__ == "__main__":
    main()
