import os
from core.samplers import EulerASampler, DDIMSampler, PLMSSampler
from modules.clip import CLIPEmbedder, EmbeddingBuilder
from modules.latent_decoder import LatentDecoder
from data import ModelLayers
from modules.stable_diffusion import StableDiffusion
import unet_mapping

import gc
import torch

def save_images(samples):
    base_count = len(os.listdir("outputs"))
    for sample in samples:
        sample.save(os.path.join("outputs", f"{base_count:05}.png"))
        base_count += 1

def main():
    wd = ModelLayers.load(unet_mapping.map["v1-5-pruned-emaonly"])
    model = StableDiffusion().half()
    model.diffusion_model.load_state_dict(wd)
    model.diffusion_model.eval()
    model.eval().cuda()

    decoder = LatentDecoder().cuda()

    seed = 45

    clip = CLIPEmbedder()
    empty_prompt = clip([""])

    prompt = EmbeddingBuilder(clip)
    prompt.add_prompt("award winning photo of a ")
    prompt.add_combined_prompt(
        ["shark", "dragon", "cat"],
        [1.5, 1, 1]
    )
    # prompt.add_prompt("in the ocean")
    prompt.add_prompt("in a lush forest")
    prompt.add_prompt(", by national geographic", weight = 1.3)

    for i in range(10):
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
