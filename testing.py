import os
from core.samplers import EulerASampler, DDIMSampler, PLMSSampler
from modules.clip import CLIPEmbedder, EmbeddingBuilder
from modules.vae_decoder import VAEDecoder
from data import ModelLayers, StableDiffusionModelData
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
    # wd = ModelLayers.load(unet_mapping.map["v1-5-pruned-emaonly"])
    print("Loading checkpoint")
    sd = StableDiffusionModelData().load_checkpoint("import/checkpoints/sd-v1-4.ckpt")

    print("Creating UNet")
    model = StableDiffusion(sd.unet_layers, use_fp16=True, device="cuda")
    model.cuda()
    # model.to(memory_format=torch.channels_last)    

    print("Creating VAE Decoder")
    decoder = VAEDecoder(sd.vae_decoder_layers)

    seed = 45

    print("Creating CLIP Embedder")
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

    print("Sampling")
    for i in range(6):
        samples = EulerASampler(model).sample(
            seed,
            512, 512, 1,
            prompt.embedding, 
            empty_prompt, 7.5, 20
        )

        print(samples.dtype)

        images = decoder(samples.type(torch.float32))

        save_images(images)

if __name__ == "__main__":
    main()
