import json
import os
from core import samplers
from core.modules.clip import OpenCLIPEmbedder, PromptBuilder, CLIPEmbedder
from core.modules.vae_decoder import VAEDecoder
from core.modules.stable_diffusion_v2 import StableDiffusion
from core.modules.vae_encoder import VAEEncoder
import models
from torch import autocast
from PIL import Image
from PIL.ExifTags import TAGS
from diffusers import StableDiffusionLatentUpscalePipeline
import torch
import requests

def save_images(images, filename, comment):
    directory = "outputs/Textures"
    if not os.path.exists(directory):
        os.makedirs(directory)    
    base_count = len(os.listdir(directory))
    filenames = []
    for image in images:
        exif_data = image.getexif()
        exif_data[0x9286] = comment
        filename = f"{filename} {base_count:05}.png"
        image.save(os.path.join(directory, filename), exif=exif_data)
        base_count += 1

        filenames.append(filename)

    requests.post("http://127.0.0.1:8000/add_images", json=filenames)       

def get_upscaler():
    old_init = torch.nn.Conv2d.__init__
    def __init__(self, *args, **kwargs):
        kwargs["padding_mode"] = "circular"
        return old_init(self, *args, **kwargs)

    torch.nn.Conv2d.__init__ = __init__

    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained("data/sd-x2-latent-upscaler", torch_dtype=torch.float16)
    upscaler.to("cuda")

    torch.nn.Conv2d.__init__ = old_init
    return upscaler

def main():
    with autocast("cuda"):
        model = StableDiffusion(1, models.unet.SD1_5_fp32, parameterization="eps", tiling=True, use_fp16=True, device="cuda").cuda().half()
        decoder = VAEDecoder(models.decoder.VAEDec1_5mse_fp32, tiling=True).cuda()
        clip = CLIPEmbedder().cuda()
        negative_prompt = "glare, imperfections, ugly, cartoon, drawn, blurry, face, diagonal"
        cfg = 7.0
        steps = 25
        negative_prompt_embedding = clip(negative_prompt)
        batch_size = 1

        while True:
            # Make http request to get list of textures
            print("Getting next material")
            response = requests.get("http://127.0.0.1:8000/next_material")
            material = response.text

            print(material)
            prompt = material + " texture, photograph, 4k, flat"
            prompt_embedding = clip(prompt)
            comment = json.dumps({"Prompt": prompt, "Negative Prompt": negative_prompt, "CFG": cfg, "Steps": steps, "Material": material, "Model": "Stable Diffusion 1.5"})

            # Run the denoising, creating the image (in latent space)
            latents = samplers.EulerASampler(model).sample(
                prompt_embedding, 
                negative_prompt_embedding,
                1024,
                1024,
                cfg=cfg,
                steps=steps,
                batch_size=batch_size
            )

            print("Decoding")
            images = decoder(latents)

            print("Saving")
            save_images(images, material, comment)

            # print("Decoding")
            # upscaled_images = decoder(upscaled_latents)

            # print("Saving")
            # save_images(upscaled_images, material, comment)

            # upscaled_latents = upscaler(
            #     prompt=prompt,
            #     image=latents,
            #     num_inference_steps=20,
            #     guidance_scale=0,
            # )




if __name__ == "__main__":
    main()
