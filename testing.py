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
from modules.clip import CLIPEmbedder, EmbeddingBuilder
from modules.latent_decoder import LatentDecoder
from data import StableDiffusionModelData
from modules.stable_diffusion import StableDiffusion

def save_images(samples):
    base_count = len(os.listdir("outputs"))
    for sample in samples:
        sample.save(os.path.join("outputs", f"{base_count:05}.png"))
        base_count += 1

def main():
    clip = CLIPEmbedder()

    empty_prompt = clip([""])

    # base = StableDiffusionModelData().load_checkpoint("data/checkpoints/Mixed.ckpt")
    base = StableDiffusionModelData().load_checkpoint("data/checkpoints/v1-5-pruned-emaonly.ckpt")
    wd = StableDiffusionModelData().load_checkpoint("data/checkpoints/wd-v1-3-float32.ckpt")

    model = StableDiffusion()
    decoder = LatentDecoder().cuda()

    seed = 45

    for i in range(10):
        combined = base.unet_layers.merge(wd.unet_layers, 0.1 * i)
        model.diffusion_model.load_state_dict(combined)
        model.cuda()

        prompt.add_prompt("1girl, black eyes, black hair, black sweater, blue background, bob cut, closed mouth, glasses, medium hair, red-framed eyewear, simple background, solo, sweater, upper body, wide-eyed")

        prompt = EmbeddingBuilder(clip)
        prompt.add_prompt("award winning photo of a ")
        prompt.add_combined_prompt(
            ["shark", "dragon", "cat"],
            [1.5, 1, 1]
        )
        # prompt.add_prompt("in the ocean")
        prompt.add_prompt("in a lush forest")
        prompt.add_prompt(", by national geographic", weight = 1.3)

        samples = EulerASampler(model).sample(
            seed,
            512, 512, 1,
            prompt.embedding, 
            empty_prompt, 7.5, 20
        )

        images = decoder.latents_to_images(samples)

        save_images(images)

if __name__ == "__main__":
    main()
