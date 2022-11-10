import os
from core.samplers import EulerASampler, DDIMSampler, PLMSSampler
from core.modules.clip import CLIPEmbedder, PromptBuilder
from core.modules.vae_decoder import VAEDecoder
from core.data import ModelLayers
from core.modules.stable_diffusion import StableDiffusion
import model_map

import gc
import torch

def save_images(samples):
    base_count = len(os.listdir("outputs"))
    for sample in samples:
        sample.save(os.path.join("outputs", f"{base_count:05}.png"))
        base_count += 1

def main():
    print("Loading UNets")
    # Loading the models to GPU
    # This speeds up merging models on the fly a lot (near instant), but uses more VRAM
    sd_unet = ModelLayers.load("SD1.5-fp32", device="cuda").half_()
    print("Loaded", sd_unet.info.name)
    wd_unet = ModelLayers.load("WD1.3-fp16", device="cuda")
    print("Loaded", wd_unet.info.name)

    print("Creating UNet")
    # Creating the UNet model in fp16 precision directly on the GPU
    # This is MUCH faster than creating it on the GPU and then moving it
    model = StableDiffusion(None, use_fp16=True, device="cuda")

    print("Creating VAE Decoder")
    decoder = VAEDecoder(ModelLayers.load("VAEDec1.4-fp32")).cuda()

    seed = 42

    print("Creating CLIP Embedder")
    clip = CLIPEmbedder().cuda()
    # Using an empty negative prompt, but it's possible to create complex ones with PromptBuilder
    negative_prompt = clip([""])

    print("Sampling")
    for i in range(6):
        # Creating a prompt with dynamic weights
        prompt = PromptBuilder(clip)
        prompt.add_prompt("award winning photo of a ")
        prompt.add_combined_prompt(
            ["shark", "dragon", "cat"],
            [0.9 + (i/10), 1, 1]
        )
        prompt.add_prompt("in a lush forest")
        prompt.add_prompt(", by national geographic", weight = 1.3)

        # Interpolate between models
        merged = sd_unet.merge(wd_unet, i/5)
        model.set(merged)

        # Run the denoising, creating the image (in latent space)
        sample = EulerASampler(model).sample(
            seed,
            512, 512, 1,
            prompt, 
            negative_prompt, 7.5, 20
        )

        # Decode the images from latent space
        images = decoder(sample)

        # Save the images
        save_images(images)

if __name__ == "__main__":
    main()
