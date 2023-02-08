import os
from core import samplers
from core.modules.clip import OpenCLIPEmbedder, PromptBuilder, FrozenOpenCLIPEmbedder
from core.modules.vae_decoder import VAEDecoder
from core.data import ModelData
from core.modules.stable_diffusion import StableDiffusion
import models
import torch

def save_images(samples, directory="test"):
    material = directory
    directory = "outputs/" + directory
    if not os.path.exists(directory):
        os.makedirs(directory)    
    base_count = len(os.listdir(directory))
    for sample in samples:
        sample.save(os.path.join(directory, f"{material} {base_count:05}.png"))
        base_count += 1

def main():
    print("Creating OpenCLIP Embedder")

    with torch.autocast("cuda"):
        clip = OpenCLIPEmbedder().cuda()
        prompt = clip.encode(["a professional photograph of an astronaut riding a triceratops"])
        print(prompt)
        negative_prompt = clip("")

    print("Creating UNet")
    unet = ModelData.load(models.unet.stable_diffusion_v2_1_512_ema_pruned_ckpt, device="cuda").half_()
    model = StableDiffusion(unet, tiling=False, use_fp16=True, context_dim=1024, use_linear_in_transformer=True, device="cuda").cuda()
    del unet

    print("Creating VAE Decoder")
    decoder = VAEDecoder(models.decoder.VAEDec1_5mse_fp32).cuda()

    print("Sampling")
    for i in range(10):
        # Run the denoising, creating the image (in latent space)
        sample = samplers.DDIMSampler(model).sample(
            prompt, 
            negative_prompt,
            512,
            512,
            steps=20
        )

        # Decode the images from latent space
        images = decoder(sample)

        # Save the images
        save_images(images)

if __name__ == "__main__":
    main()
