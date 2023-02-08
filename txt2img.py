import os
from core import samplers
from core.modules.clip import CLIPEmbedder, PromptBuilder
from core.modules.vae_decoder import VAEDecoder
from core.data import ModelData
from core.modules.stable_diffusion import StableDiffusion
import models
import torch

def save_images(samples, directory):
    material = directory
    directory = "outputs/" + directory
    if not os.path.exists(directory):
        os.makedirs(directory)    
    base_count = len(os.listdir(directory))
    for sample in samples:
        sample.save(os.path.join(directory, f"{material} {base_count:05}.png"))
        base_count += 1

def main():
    print("Creating UNet")
    unet = ModelData.load(models.unet.SD1_5_fp32, device="cuda").half_()
    model = StableDiffusion(unet, tiling=True, use_fp16=True, device="cuda").cuda()
    
    # def tiling(el):
    #     for child in el.children():
    #         tiling(child)
    #         if type(child) == torch.nn.Conv2d:
    #             print(child.padding_mode)
    #             child.padding_mode = "circular"
    
    # tiling(model)

    del unet

    print("Creating VAE Decoder")
    decoder = VAEDecoder(models.decoder.VAEDec1_5mse_fp32).cuda()

    print("Creating CLIP Embedder")
    clip = CLIPEmbedder().cuda()
    # prompt = clip("a beautiful empress portrait, with a brilliant, impossible striking big Cat headpiece, clothes made of cats, everything cats, symmetrical, dramatic studio lighting, rococo, baroque, greens, asian, hyperrealism, closeup, D&D, fantasy, intricate, elegant, highly detailed, digital painting, artstation, octane render, 8k, concept art, matte, sharp focus, illustration, art by Artgerm and Greg Rutkowski and Alphonse Mucha")
    # prompt = clip("red vibe gundam mobile suite armor in the battlefield, mar planet, futuristic style, intricate, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, unreal engine 5, 8 k, art by artgerm and greg rutkowski and alphonse mucha")
    # prompt = clip("full body pose, hyperrealistic photograph of baby yoda dressed as the mad hatter, dim volumetric lighting, 8 k, octane beautifully detailed render, extremely hyper detailed, intricate, epic composition, cinematic lighting, masterpiece, trending on artstation, very very detailed, stunning, hdr, smooth, sharp focus, high resolution, award, winning photo, dslr, 5 0 mm")
    # prompt = clip("real life photo of a beautiful girl, full body photoshoot, long brown hair, brown eyes, full round face, short smile, wool sweater belly free, forest setting, cinematic lightning, medium shot, mid - shot, highly detailed, trending on artstation, unreal engine 4 k, 8 0 mm, 8 5 mm, cinematic wallpaper")
    # prompt = clip("Michael Fassbender in white cloak, intricate, body portrait, epic lighting, hyper realistic,ray tracing, white short hair, character concept art, cinematic, artgerm, ultra detailed, artstation trending.")
    # prompt = clip("portrait skull girl astronaut by petros afshar, tom whalen, mucha, laurie greasley, war face by greg rutkowski ")

    # negative_prompt = clip(["ugly, duplicate, morbid, mutilated, out of frame, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, disfigured, out of frame, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"])
    negative_prompt = clip("")

    seed = 42

    print("Sampling")

    for material in ["Emperador gold marble"]:
        for i in range(10):
            builder = PromptBuilder(clip)
            builder.add_prompt(material + ", 4k, photorealistic")

            # Run the denoising, creating the image (in latent space)
            sample = samplers.DPMpp2MKarrasSampler(model).sample(
                builder, 
                negative_prompt,
                1024,
                1024,
                steps=20
            )

            # Decode the images from latent space
            images = decoder(sample)

            # Save the images
            save_images(images, material)

if __name__ == "__main__":
    main()
