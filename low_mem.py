import os
from core.samplers import EulerASampler, DDIMSampler, PLMSSampler
from core.modules.clip import CLIPEmbedder, PromptBuilder
from core.modules.vae_decoder import VAEDecoder
from core.data import ModelData
from core.modules.stable_diffusion import StableDiffusion
import models

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
    sd_unet = ModelData.load(models.unet.SD1_5_fp32, device="cpu").half_()
    print("Loaded", sd_unet.info.name)
    wd_unet = ModelData.load(models.unet.WD1_3_fp32, device="cpu").half_()
    print("Loaded", wd_unet.info.name)

    print("Creating UNet")
    # Creating the UNet model in fp16 precision directly on the GPU
    # This is MUCH faster than creating it on the CPU and then moving it
    model = StableDiffusion(None, use_fp16=True, device="cuda")
    model.cpu()

    print("Creating VAE Decoder")
    decoder = VAEDecoder(ModelData.load(models.decoder.VAEDec1_4_fp32))

    seed = 42

    print("Creating CLIP Embedder")
    clip = CLIPEmbedder()
    # Using an empty negative prompt, but it's possible to create complex ones with PromptBuilder
    negative_prompt = clip([""])
    simple = clip("award winning photo of a cat in a lush forest")

    print("Sampling")
    for i in range(6):
        # Creating a prompt with dynamic weights
        print("Creating prompt")
        prompt = PromptBuilder(clip)
        prompt.add_prompt("award winning photo of a cat in a lush forest")
        # prompt.add_combined_prompt(
        #     ["shark", "dragon", "cat"],
        #     [0.9 + (i/10), 1, 1]
        # )
        # prompt.add_prompt("in a lush forest")
        # prompt.add_prompt(", by national geographic", weight = 1.3)

        # Interpolate between models
        print("Merging models")
        print("Moving model to cpu")
        model.cpu()
        print("Moving sd and wd to GPU")
        sd_unet.cuda()
        wd_unet.cuda()

        print("Merging")
        merged = sd_unet.merge(wd_unet, i/5)

        print("Moving sd & wd to cpu")
        sd_unet.cpu()
        wd_unet.cpu()

        print("Moving model to GPU")
        model.cuda()
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

        # Run the denoising, creating the image (in latent space)
        sample = EulerASampler(model).sample(
            seed,
            512, 512, 1,
            simple, 
            negative_prompt, 7.5, 20
        )

        # Decode the images from latent space
        images = decoder(sample)

        # Save the images
        save_images(images)

if __name__ == "__main__":
    main()
