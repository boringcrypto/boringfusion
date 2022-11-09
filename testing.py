import os
from core.samplers import EulerASampler, DDIMSampler, PLMSSampler
from modules.clip import CLIPEmbedder, EmbeddingBuilder
from modules.vae_decoder import VAEDecoder
from core.data import ModelLayers
from modules.stable_diffusion import StableDiffusion
import model_map

import gc
import torch

def save_images(samples):
    base_count = len(os.listdir("outputs"))
    for sample in samples:
        sample.save(os.path.join("outputs", f"{base_count:05}.png"))
        base_count += 1

def main():
    print("Loading UNet")
    unet = ModelLayers.load(model_map.map("Stable Diffusion v1.5 EMA fp32"))

    print("Creating UNet")
    model = StableDiffusion(unet, use_fp16=True, device="cuda")
    # model.to(memory_format=torch.channels_last)    

    print("Creating VAE Decoder")
    decoder_layers = ModelLayers.load(model_map.map["8b7877f3"])
    decoder = VAEDecoder(decoder_layers)

    seed = 45

    print("Creating CLIP Embedder")
    clip = CLIPEmbedder()
    empty_prompt = clip([""])

    print("Sampling")
    for i in range(6):
        prompt = EmbeddingBuilder(clip)
        prompt.add_prompt("award winning photo of a ")
        prompt.add_combined_prompt(
            ["shark", "dragon", "cat"],
            [1.5 * (i/6), 1, 1]
        )
        # prompt.add_prompt("in the ocean")
        prompt.add_prompt("in a lush forest")
        prompt.add_prompt(", by national geographic", weight = 1.3)

        sample = EulerASampler(model).sample(
            seed + 1,
            512, 512, 1,
            prompt.embedding, 
            empty_prompt, 7.5, 20
        )

        images = decoder(sample.type(torch.float32))
        save_images(images)

if __name__ == "__main__":
    main()
