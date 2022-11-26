import os
from core import samplers
from core.modules.clip import CLIPEmbedder, PromptBuilder
from core.modules.vae_decoder import VAEDecoder
from core.data import ModelData
from core.modules.stable_diffusion import StableDiffusion
import models

def save_images(samples):
    base_count = len(os.listdir("outputs"))
    for sample in samples:
        sample.save(os.path.join("outputs", f"{base_count:05}.png"))
        base_count += 1

def main():
    print("Creating UNet")
    unet = ModelData.load(models.unet.f222_fp32, device="cuda").half_()
    model = StableDiffusion(unet, use_fp16=True, device="cuda").cuda()
    del unet

    print("Creating VAE Decoder")
    decoder = VAEDecoder(models.decoder.VAEDec1_5mse_fp32)

    print("Creating CLIP Embedder")
    clip = CLIPEmbedder()
    # prompt = clip("a beautiful empress portrait, with a brilliant, impossible striking big Cat headpiece, clothes made of cats, everything cats, symmetrical, dramatic studio lighting, rococo, baroque, greens, asian, hyperrealism, closeup, D&D, fantasy, intricate, elegant, highly detailed, digital painting, artstation, octane render, 8k, concept art, matte, sharp focus, illustration, art by Artgerm and Greg Rutkowski and Alphonse Mucha")
    # prompt = clip("red vibe gundam mobile suite armor in the battlefield, mar planet, futuristic style, intricate, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, unreal engine 5, 8 k, art by artgerm and greg rutkowski and alphonse mucha")
    # prompt = clip("full body pose, hyperrealistic photograph of baby yoda dressed as the mad hatter, dim volumetric lighting, 8 k, octane beautifully detailed render, extremely hyper detailed, intricate, epic composition, cinematic lighting, masterpiece, trending on artstation, very very detailed, stunning, hdr, smooth, sharp focus, high resolution, award, winning photo, dslr, 5 0 mm")
    # prompt = clip("real life photo of a beautiful girl, full body photoshoot, long brown hair, brown eyes, full round face, short smile, wool sweater belly free, forest setting, cinematic lightning, medium shot, mid - shot, highly detailed, trending on artstation, unreal engine 4 k, 8 0 mm, 8 5 mm, cinematic wallpaper")
    # prompt = clip("Michael Fassbender in white cloak, intricate, body portrait, epic lighting, hyper realistic,ray tracing, white short hair, character concept art, cinematic, artgerm, ultra detailed, artstation trending.")
    # prompt = clip("portrait skull girl astronaut by petros afshar, tom whalen, mucha, laurie greasley, war face by greg rutkowski ")
    builder = PromptBuilder(clip)
    builder.add_prompt("highly detailed photography, spring, portrait of beautiful naked woman sitting on the altar of a cathedral, ")
    builder.add_prompt(", wearing a hajib, perfect face, perfect body, large breasts, trimmed pussy, vagina, spread legs")
    builder.add_prompt("sharp focus", 1.25)
    builder.add_prompt(", photo by Annie Leibovitz, metart, intricate, natural lighting, high quality, 4k, cover photo")
    
    negative_prompt = clip(["out of focus body, out of focus face, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, out of frame, ugly, extra limbs, (bad anatomy), gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, fused fingers, too many fingers, long neck"])

    seed = 42

    print("Sampling")
    for i in range(10):
        # Run the denoising, creating the image (in latent space)
        sample = samplers.DPMpp2SaKarrasSampler(model).sample(
            seed + i,
            512, 512, 1,
            builder, 
            negative_prompt, 7.5, 8
        )

        # Decode the images from latent space
        images = decoder(sample)

        # Save the images
        save_images(images)

if __name__ == "__main__":
    main()
